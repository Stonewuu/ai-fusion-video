package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.mapper.ai.AgentUserOverrideMapper;
import com.stonewu.fusion.service.ai.configuration.AgentUserOverrideRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class AgentUserConfigurationIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container
    static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "agent-config-" + UUID.randomUUID()).toAbsolutePath();
    static final AtomicLong IDS = new AtomicLong(89000);

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry p) {
        p.add("spring.datasource.url", MYSQL::getJdbcUrl);
        p.add("spring.datasource.username", MYSQL::getUsername);
        p.add("spring.datasource.password", MYSQL::getPassword);
        p.add("spring.data.redis.host", REDIS::getHost);
        p.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        p.add("spring.data.redis.password", () -> "");
        p.add("app.storage.local-base-path", FILES::toString);
        p.add("app.agent-workspace.local-base-path", () -> FILES.resolve("workspace").toString());
    }

    @Autowired
    TestRestTemplate http;
    @Autowired
    TokenService tokens;
    @Autowired
    JdbcTemplate jdbc;
    @Autowired
    AgentUserOverrideMapper overrideMapper;
    final String endpoint = "/api/ai/agent-config/definitions/episode_storyboard_writer";

    @Test
    void userOverridesArePersistedCachedIsolatedAndResetWithOptimisticConcurrency() {
        var first = user();
        var other = user();
        var initial = call(HttpMethod.GET, endpoint, null, first).path("data").path("configuration");
        assertThat(initial.path("revision").asLong()).isZero();
        var overrides = Map.of("name", "我的分镜编剧", "systemPrompt", "镜头之间保持人物与光线连续。", "toolNames",
                List.of(), "maxIters", 0);
        var saved = call(HttpMethod.PUT, endpoint, Map.of("expectedRevision", 0, "overrides", overrides), first);
        assertThat(saved.path("code").asInt()).isZero();
        assertThat(saved.path("data").path("configuration").path("effective").path("name").asText()).isEqualTo(
                "我的分镜编剧");
        // 第二次 HTTP 读走缓存；另一个账号始终读取自己的配置。
        assertThat(call(HttpMethod.GET, endpoint, null, first).path("data").path("configuration").path("revision")
                .asLong()).isEqualTo(1);
        assertThat(call(HttpMethod.GET, endpoint, null, other).path("data").path("configuration").path("effective")
                .path("name")).isEqualTo(initial.path("defaults").path("name"));
        assertThat(
                call(HttpMethod.PUT, endpoint, Map.of("expectedRevision", 0, "overrides", Map.of("name", "过期编辑")),
                        first).path("code").asInt()).isEqualTo(409);
        var reset = call(HttpMethod.PUT, endpoint, Map.of("expectedRevision", 1, "overrides", Map.of()), first);
        assertThat(reset.path("code").asInt()).isZero();
        assertThat(reset.path("data").path("configuration").path("effective")).isEqualTo(initial.path("defaults"));
        var row = jdbc.queryForMap(
                "SELECT user_id,revision,overrides_json FROM afv_agent_user_override WHERE agent_type='episode_storyboard_writer' AND revision=2");
        assertThat(row.get("revision")).isEqualTo(2L);
        var reopened = new AgentUserOverrideRepository(overrideMapper).get(((Number) row.get("user_id")).longValue(),
                "episode_storyboard_writer");
        assertThat(reopened.json()).isEqualTo("{}");
    }

    @Test
    void skillDefaultsAndCustomNamesReachTheCanvasCatalogAndDeletedSkillsFailClearly() {
        var headers = user();
        var skill = call(HttpMethod.PUT, "/api/ai/agent-config/skills",
                Map.of("name", "shot-review", "displayName", "镜头检查", "description", "检查连续性", "content",
                        "检查每个镜头中的人物、服装和光线。"), headers);
        assertThat(skill.path("code").asInt()).isZero();
        var saved = call(HttpMethod.PUT, endpoint, Map.of("expectedRevision", 0, "overrides",
                Map.of("name", "定制分镜", "defaultSkills", List.of("shot-review"))), headers);
        assertThat(saved.path("code").asInt()).isZero();
        var catalog = call(HttpMethod.GET, "/api/canvases/capabilities", null, headers);
        assertThat(catalog.path("code").asInt()).isZero();
        var profiles = catalog.path("data").path("agentProfiles");
        boolean found = false;
        for (var profile : profiles) {
            if (profile.path("type").asText().equals("episode_storyboard_writer")) {
                found = true;
                assertThat(profile.path("name").asText()).isEqualTo("定制分镜");
                assertThat(profile.path("defaultSkills").get(0).asText()).isEqualTo("shot-review");
                assertThat(profile.path("customized").asBoolean()).isTrue();
            }
        }
        assertThat(found).isTrue();
        assertThat(catalog.path("data").path("skills").toString()).contains("镜头检查", "shot-review");
        assertThat(call(HttpMethod.DELETE, "/api/ai/agent-config/skills/shot-review", null, headers).path("code")
                .asInt()).isZero();
        var invalid = call(HttpMethod.PUT, endpoint,
                Map.of("expectedRevision", 1, "overrides", Map.of("defaultSkills", List.of("shot-review"))), headers);
        assertThat(invalid.path("code").asInt()).isEqualTo(400);
        assertThat(invalid.path("msg").asText()).contains("Skill 不可用");
    }

    @Test
    void rejectsUnknownToolsNegativeLimitsAndBlankPromptsWithoutPersisting() {
        var headers = user();
        for (var overrides : List.of(Map.of("toolNames", List.of("execute_arbitrary_command")), Map.of("maxIters", -1),
                Map.of("systemPrompt", "   "))) {
            assertThat(
                    call(HttpMethod.PUT, endpoint, Map.of("expectedRevision", 0, "overrides", overrides), headers).path(
                            "code").asInt()).isEqualTo(400);
        }
        assertThat(call(HttpMethod.GET, endpoint, null, headers).path("data").path("configuration").path("revision")
                .asLong()).isZero();
    }

    HttpHeaders user() {
        long id = IDS.incrementAndGet();
        String name = "agent-config-" + id;
        jdbc.update("INSERT INTO sys_user(id,username,password) VALUES(?,?,'test')", id, name);
        var headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(tokens.createToken(id, name, null).getAccessToken());
        return headers;
    }

    JsonNode call(HttpMethod method, String path, Object body, HttpHeaders headers) {
        var response = http.exchange(path, method, new HttpEntity<>(body, headers), JsonNode.class);
        assertThat(response.getBody()).isNotNull();
        return response.getBody();
    }
}
