package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.system.Role;
import com.stonewu.fusion.entity.system.User;
import com.stonewu.fusion.entity.team.Team;
import com.stonewu.fusion.entity.team.TeamMember;
import com.stonewu.fusion.mapper.ai.ApiConfigMapper;
import com.stonewu.fusion.mapper.system.RoleMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.mapper.team.TeamMapper;
import com.stonewu.fusion.mapper.team.TeamMemberMapper;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.system.SystemConfigService;
import com.stonewu.fusion.service.system.UserService;
import com.stonewu.fusion.service.system.RoleService;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.cache.CacheManager;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;

/** 真实 HTTP、MySQL、Redis 验证对象越权、团队提权、配置泄漏和禁用账户，不调用外部供应商。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ObjectAuthorizationIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container
    static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "object-authorization-" + UUID.randomUUID()).toAbsolutePath();

    @DynamicPropertySource
    static void properties(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url", MYSQL::getJdbcUrl);
        properties.add("spring.datasource.username", MYSQL::getUsername);
        properties.add("spring.datasource.password", MYSQL::getPassword);
        properties.add("spring.data.redis.host", REDIS::getHost);
        properties.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        properties.add("spring.data.redis.password", () -> "");
        properties.add("app.storage.local-base-path", FILES::toString);
        properties.add("app.agent-workspace.local-base-path", () -> FILES.resolve("workspace").toString());
    }

    @Autowired TestRestTemplate http;
    @Autowired TokenService tokens;
    @Autowired UserMapper users;
    @Autowired TeamMapper teams;
    @Autowired TeamMemberMapper members;
    @Autowired RoleMapper roles;
    @Autowired UserService userService;
    @Autowired RoleService roleService;
    @Autowired ApiConfigMapper apiConfigs;
    @Autowired SystemConfigService configs;
    @Autowired CacheManager caches;

    record Account(User user, TokenService.TokenPair tokens, HttpHeaders headers) {
    }

    record Fixture(Account owner, Account outsider, long project, long asset, long item,
                   long script, long episode, long scene, long storyboard) {
    }

    @Test
    @Order(1)
    void concurrentPublicInitializationCreatesExactlyOneAdministrator() {
        var start = new CountDownLatch(1);
        var calls = IntStream.range(0, 6).mapToObj(index -> CompletableFuture.supplyAsync(() -> {
            try {
                start.await();
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(interrupted);
            }
            return http.postForEntity("/api/system/init/setup", Map.of("username", "bootstrap-" + index,
                    "password", "only-for-isolated-tests", "nickname", "初始化测试"), JsonNode.class);
        })).toList();
        start.countDown();
        var statuses = calls.stream().map(CompletableFuture::join).map(ResponseEntity::getStatusCode).toList();
        assertThat(statuses).filteredOn(HttpStatus.OK::equals).hasSize(1);
        assertThat(statuses).filteredOn(HttpStatus.BAD_REQUEST::equals).hasSize(5);
        assertThat(users.selectCount(Wrappers.<User>lambdaQuery().likeRight(User::getUsername, "bootstrap-"))).isEqualTo(1);
    }

    @Test
    @Order(2)
    void publicRegistrationCanOnlyJoinTheDefaultTeamAsAnOrdinaryMember() {
        configs.setRegistrationEnabled(true);
        try {
            var registered = ok(http.postForEntity("/api/auth/register", Map.of("username", "registered-user",
                    "password", "only-for-isolated-tests", "confirmPassword", "only-for-isolated-tests"), JsonNode.class));
            TeamMember member = members.selectOne(Wrappers.<TeamMember>lambdaQuery()
                    .eq(TeamMember::getUserId, registered.path("userId").asLong()));
            assertThat(member.getRole()).isEqualTo(3);
            assertThat(member.getTeamId()).isEqualTo(teams.selectList(null).getFirst().getId());
        } finally {
            configs.setRegistrationEnabled(false);
        }
    }

    @Test
    void disablingARoleRevokesCachedAuthoritiesFromAnExistingSession() {
        Account admin = account();
        Role role = roles.selectOne(Wrappers.<Role>lambdaQuery().eq(Role::getCode, "admin"));
        userService.assignRole(admin.user().getId(), role.getId());
        ok(get("/api/system/config", admin));
        roleService.updateRole(role.getId(), null, null, 0, null);
        try {
            // 模拟撤权前的并发读取在缓存清理后才回填旧角色。
            caches.getCache("userRoles").put(admin.user().getId(), List.of(role));
            forbidden(get("/api/system/config", admin));
            assertThat(ok(get("/api/auth/user-info", admin)).path("roles")).isEmpty();
        } finally {
            roleService.updateRole(role.getId(), null, null, 1, null);
        }
        ok(get("/api/system/config", admin));
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void unrelatedPersonalUsersAndTeamsCannotReadOrMutateAnyReportedObject(boolean teamOwned) {
        var f = fixture(teamOwned);
        var paths = List.of("/api/project/" + f.project(),
                "/api/project/" + f.project() + "/workspace-overview",
                "/api/project/" + f.project() + "/members",
                "/api/asset/" + f.asset(), "/api/asset/item/" + f.item(),
                "/api/asset/" + f.asset() + "/items",
                "/api/asset/list?projectId=" + f.project(),
                "/api/asset/list-with-items?projectId=" + f.project(),
                "/api/asset/all?projectId=" + f.project(),
                "/api/script/" + f.script(), "/api/script/project/" + f.project(),
                "/api/script/" + f.script() + "/episodes", "/api/script/episode/" + f.episode(),
                "/api/script/episode/" + f.episode() + "/scenes", "/api/script/scene/" + f.scene(),
                "/api/storyboard/" + f.storyboard(), "/api/storyboard/project/" + f.project(),
                "/api/storyboard/" + f.storyboard() + "/statistics");
        var before = new LinkedHashMap<String, JsonNode>();
        for (String path : paths) {
            before.put(path, ok(get(path, f.owner())));
            forbidden(get(path, f.outsider()));
        }

        forbidden(send(HttpMethod.PUT, "/api/project", patch(f.project(), "name"), f.outsider()));
        forbidden(send(HttpMethod.PUT, "/api/asset", patch(f.asset(), "name"), f.outsider()));
        var itemPatch = patch(f.item(), "name");
        itemPatch.put("assetVersion", 0);
        forbidden(send(HttpMethod.PUT, "/api/asset/item", itemPatch, f.outsider()));
        forbidden(send(HttpMethod.PUT, "/api/script", patch(f.script(), "content"), f.outsider()));
        forbidden(send(HttpMethod.PUT, "/api/script/episode", patch(f.episode(), "title"), f.outsider()));
        forbidden(send(HttpMethod.PUT, "/api/script/scene", patch(f.scene(), "sceneDescription"), f.outsider()));
        forbidden(send(HttpMethod.PUT, "/api/script/" + f.script() + "/source",
                Map.of("requestId", requestId(), "version", 0, "rawContent", "unauthorized"), f.outsider()));
        forbidden(send(HttpMethod.PUT, "/api/storyboard", patch(f.storyboard(), "description"), f.outsider()));
        forbidden(send(HttpMethod.POST, "/api/asset", Map.of("requestId", requestId(), "projectId", f.project(),
                "fields", Map.of("name", "injected", "type", "character")), f.outsider()));
        forbidden(send(HttpMethod.POST, "/api/asset/item", Map.of("requestId", requestId(), "assetId", f.asset(),
                "parentVersion", 0, "fields", Map.of("name", "injected")), f.outsider()));
        forbidden(send(HttpMethod.POST, "/api/script/episode", child(f.script(), 0, Map.of("title", "injected")),
                f.outsider()));
        forbidden(send(HttpMethod.POST, "/api/script/scene", child(f.episode(), 0, Map.of("sceneDescription", "injected")),
                f.outsider()));
        forbidden(send(HttpMethod.POST, "/api/project/" + f.project() + "/members",
                membership(f.outsider(), 0, 2), f.outsider()));
        forbidden(send(HttpMethod.POST, "/api/project/" + f.project() + "/workspace/initialize",
                Map.of("requestId", requestId(), "version", 0), f.outsider()));
        for (String path : List.of("/api/project/" + f.project(), "/api/asset/" + f.asset(),
                "/api/asset/item/" + f.item(), "/api/script/episode/" + f.episode(),
                "/api/script/scene/" + f.scene())) {
            forbidden(send(HttpMethod.DELETE, path + "?version=0&parentVersion=0&requestId=" + requestId(),
                    null, f.outsider()));
        }
        for (String path : paths) {
            assertThat(ok(get(path, f.owner()))).as(path + " remains unchanged").isEqualTo(before.get(path));
        }
        for (String path : List.of("/api/project/list", "/api/project/page", "/api/asset/all",
                "/api/asset/list-with-items")) {
            assertThat(ok(get(path, f.outsider())).toString()).doesNotContain("private-" + f.project());
        }
    }

    @Test
    void viewerCanReadButCannotWriteAndRevocationAlsoBlocksCachedReadsAndCommandReplay() {
        var f = fixture(false);
        ok(send(HttpMethod.POST, "/api/project/" + f.project() + "/members", membership(f.outsider(), 0, 4), f.owner()));
        ok(get("/api/asset/" + f.asset(), f.outsider()));
        forbidden(send(HttpMethod.PUT, "/api/asset", patch(f.asset(), "name"), f.outsider()));
        forbidden(send(HttpMethod.POST, "/api/project/" + f.project() + "/members",
                membership(f.outsider(), 1, 2), f.outsider()));
        ok(send(HttpMethod.POST, "/api/project/" + f.project() + "/members", membership(f.outsider(), 1, 3), f.owner()));
        var change = patch(f.asset(), "name");
        var receipt = ok(send(HttpMethod.PUT, "/api/asset", change, f.outsider()));
        assertThat(ok(send(HttpMethod.PUT, "/api/asset", change, f.outsider()))).isEqualTo(receipt);
        var removal = new LinkedHashMap<>(membership(f.outsider(), 2, 3));
        removal.put("role", null);
        ok(send(HttpMethod.POST, "/api/project/" + f.project() + "/members", removal, f.owner()));
        forbidden(get("/api/asset/" + f.asset(), f.outsider()));
        forbidden(send(HttpMethod.PUT, "/api/asset", change, f.outsider()));
        assertThat(ok(get("/api/asset/" + f.asset(), f.owner()))).isEqualTo(receipt);
    }

    @Test
    void teamManagementCannotBeUsedToJoinAnotherScopePromoteSelfOrRemoveOwner() {
        Account owner = account();
        Account member = account();
        Account outsider = account();
        long team = team(owner);
        long foreignTeam = team(outsider);
        member(team, member, 3);
        forbidden(get("/api/team/get?id=" + team, outsider));
        forbidden(get("/api/team/member/list?teamId=" + team, outsider));
        ok(get("/api/team/get?id=" + team, member));
        forbidden(send(HttpMethod.PUT, "/api/team/update", Map.of("id", team, "name", "defaced"), outsider));
        forbidden(send(HttpMethod.PUT, "/api/team/update", Map.of("id", team, "name", "defaced"), member));
        forbidden(send(HttpMethod.POST, "/api/team/member/add", Map.of("teamId", team,
                "userId", outsider.user().getId(), "role", 2), outsider));
        forbidden(send(HttpMethod.POST, roleUrl(team, member, 2), null, member));
        forbidden(send(HttpMethod.POST, "/api/team/member/remove?teamId=" + team + "&userId=" + owner.user().getId(),
                null, member));
        forbidden(send(HttpMethod.POST, "/api/team/member/add", Map.of("teamId", foreignTeam,
                "userId", member.user().getId(), "role", 2), owner));
        assertThat(members.selectCount(Wrappers.<TeamMember>lambdaQuery().eq(TeamMember::getTeamId, team))).isEqualTo(2);
        assertThat(teams.selectById(team).getName()).isEqualTo("private-team");
        assertThat(send(HttpMethod.POST, roleUrl(team, member, 1), null, owner).getStatusCode())
                .isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(send(HttpMethod.POST, roleUrl(team, member, 99), null, owner).getStatusCode())
                .isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(send(HttpMethod.POST, roleUrl(team, owner, 3), null, owner).getStatusCode())
                .isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(send(HttpMethod.POST, "/api/team/member/remove?teamId=" + team + "&userId=" + owner.user().getId(),
                null, owner).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        ok(send(HttpMethod.POST, roleUrl(team, member, 2), null, owner));
        ok(send(HttpMethod.PUT, "/api/team/update", Map.of("id", team, "name", "authorized"), member));
        Account invited = account();
        ok(send(HttpMethod.POST, "/api/team/member/add", Map.of("teamId", team,
                "userId", invited.user().getId(), "role", 3), member));
        forbidden(send(HttpMethod.POST, roleUrl(team, invited, 2), null, member));
        ok(send(HttpMethod.POST, "/api/team/member/remove?teamId=" + team + "&userId=" + member.user().getId(),
                null, owner));
        forbidden(send(HttpMethod.PUT, "/api/team/update", Map.of("id", team, "name", "revoked"), member));
        assertThat(teams.selectById(team).getName()).isEqualTo("authorized");
    }

    @Test
    void removedTeamMemberCanBeReaddedWithTheNewRole() {
        Account owner = account();
        Account invited = account();
        long team = team(owner);
        String path = "/api/team/member/add";
        ok(send(HttpMethod.POST, path, Map.of("teamId", team,
                "userId", invited.user().getId(), "role", 2), owner));
        ok(send(HttpMethod.POST, "/api/team/member/remove?teamId=" + team + "&userId=" + invited.user().getId(),
                null, owner));
        long restored = ok(send(HttpMethod.POST, path, Map.of("teamId", team,
                "userId", invited.user().getId(), "role", 3), owner)).asLong();
        TeamMember member = members.selectById(restored);
        assertThat(member.getRole()).isEqualTo(3);
        assertThat(member.getStatus()).isEqualTo(1);
        assertThat(members.selectCount(Wrappers.<TeamMember>lambdaQuery().eq(TeamMember::getTeamId, team)
                .eq(TeamMember::getUserId, invited.user().getId()))).isEqualTo(1);
        ok(get("/api/team/member/list?teamId=" + team, invited));
    }

    @Test
    void ordinaryUsersCannotReadProviderOrSystemSecretsButCanUsePublicResourceConfiguration() {
        Account user = account();
        Account admin = account();
        Role role = roles.selectOne(Wrappers.<Role>lambdaQuery().eq(Role::getCode, "admin"));
        userService.assignRole(admin.user().getId(), role.getId());
        ApiConfig provider = ApiConfig.builder().name("security-fixture").platform("openai")
                .apiUrl("https://example.invalid").apiKey("test-provider-secret").appSecret("test-app-secret")
                .proxyPassword("test-proxy-secret").status(1).build();
        apiConfigs.insert(provider);
        configs.setValue("mail_password", "test-mail-secret");
        configs.setValue("security_future_secret", "test-future-secret");
        configs.setValue("resource_base_url", "https://example.invalid");
        for (String path : List.of("/api/ai/api-config/get?id=" + provider.getId(), "/api/ai/api-config/list",
                "/api/ai/api-config/page", "/api/system/config", "/api/system/config/mail_password",
                "/api/system/config/security_future_secret")) {
            forbidden(get(path, user));
            ok(get(path, admin));
            assertThat(http.getForEntity(path, JsonNode.class).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
        }
        assertThat(ok(get("/api/system/config/resource_base_url", user)).asText()).isEqualTo("https://example.invalid");
        ok(get("/api/ai/model/list", user));
        ok(get("/api/storage/config/list", user));
    }

    @Test
    void disabledAccountsLoseExistingAccessAndRefreshTokensAndSessionsCannotChangeIdentity() {
        Account user = account();
        Account another = account();
        ok(get("/api/auth/user-info", user));
        userService.updateUser(user.user().getId(), null, null, null, null, 0);
        // 模拟历史缓存尚未过期；认证必须以数据库中的禁用状态为准。
        caches.getCache("userByUsername").put(user.user().getUsername(), user.user());
        assertThat(get("/api/auth/user-info", user).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
        assertThat(send(HttpMethod.POST, "/api/auth/refresh", Map.of("refreshToken", user.tokens().getRefreshToken()),
                user).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
        var mismatched = tokens.createToken(user.user().getId(), another.user().getUsername(), null);
        var forged = new Account(user.user(), mismatched, headers(mismatched));
        assertThat(get("/api/auth/user-info", forged).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
        assertThat(send(HttpMethod.POST, "/api/auth/refresh", Map.of("refreshToken", mismatched.getRefreshToken()),
                forged).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
        ok(get("/api/auth/user-info", another));
        ok(send(HttpMethod.POST, "/api/auth/refresh", Map.of("refreshToken", another.tokens().getRefreshToken()), another));
    }

    private Fixture fixture(boolean teamOwned) {
        Account owner = account();
        Account outsider = account();
        if (teamOwned) {
            team(owner);
            team(outsider);
        }
        long project = ok(send(HttpMethod.POST, "/api/project", Map.of("requestId", requestId(),
                "fields", Map.of("name", "private-project")), owner)).path("id").asLong();
        ok(send(HttpMethod.PUT, "/api/project", Map.of("requestId", requestId(), "id", project, "version", 0,
                "patch", Map.of("name", "private-" + project)), owner));
        var asset = ok(send(HttpMethod.POST, "/api/asset", Map.of("requestId", requestId(), "projectId", project,
                "fields", Map.of("name", "private-" + project, "type", "character")), owner));
        long script = ok(get("/api/script/project/" + project, owner)).path("id").asLong();
        int scriptVersion = ok(get("/api/script/" + script, owner)).path("version").asInt();
        var episode = ok(send(HttpMethod.POST, "/api/script/episode", child(script, scriptVersion,
                Map.of("title", "private-episode", "episodeNumber", 1)), owner));
        long scene = ok(send(HttpMethod.POST, "/api/script/scene", child(episode.path("id").asLong(),
                episode.path("version").asInt(), Map.of("sceneNumber", "1", "sceneDescription", "private-scene")), owner))
                .path("id").asLong();
        long storyboard = ok(get("/api/storyboard/project/" + project, owner)).path("id").asLong();
        return new Fixture(owner, outsider, project, asset.path("id").asLong(), asset.path("items").get(0).path("id").asLong(),
                script, episode.path("id").asLong(), scene, storyboard);
    }

    private Account account() {
        User user = User.builder().username("audit-" + UUID.randomUUID()).nickname("权限测试")
                .password("unused-test-password").status(1).build();
        users.insert(user);
        var pair = tokens.createToken(user.getId(), user.getUsername(), null);
        return new Account(user, pair, headers(pair));
    }

    private long team(Account owner) {
        Team team = Team.builder().name("private-team").ownerUserId(owner.user().getId()).status(1).build();
        teams.insert(team);
        member(team.getId(), owner, 1);
        return team.getId();
    }

    private void member(long teamId, Account user, int role) {
        members.insert(TeamMember.builder().teamId(teamId).userId(user.user().getId()).role(role).status(1).build());
    }

    private HttpHeaders headers(TokenService.TokenPair pair) {
        var headers = new HttpHeaders();
        headers.setBearerAuth(pair.getAccessToken());
        headers.setContentType(MediaType.APPLICATION_JSON);
        return headers;
    }

    private Map<String, Object> membership(Account user, long revision, int role) {
        return Map.of("requestId", requestId(), "expectedRevision", revision, "userId", user.user().getId(), "role", role);
    }

    private String roleUrl(long team, Account user, int role) {
        return "/api/team/member/change-role?teamId=" + team + "&userId=" + user.user().getId() + "&role=" + role;
    }

    private Map<String, Object> child(long parent, int version, Map<String, Object> fields) {
        return Map.of("requestId", requestId(), "parentId", parent, "parentVersion", version, "fields", fields);
    }

    private LinkedHashMap<String, Object> patch(long id, String field) {
        return new LinkedHashMap<>(Map.of("requestId", requestId(), "id", id, "version", 0,
                "patch", Map.of(field, "unauthorized")));
    }

    private String requestId() {
        return UUID.randomUUID().toString();
    }

    private ResponseEntity<JsonNode> get(String path, Account user) {
        return send(HttpMethod.GET, path, null, user);
    }

    private ResponseEntity<JsonNode> send(HttpMethod method, String path, Object body, Account user) {
        return http.exchange(path, method, new HttpEntity<>(body, user.headers()), JsonNode.class);
    }

    private JsonNode ok(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull();
        assertThat(response.getBody().path("code").asInt()).as(response.toString()).isZero();
        return response.getBody().path("data");
    }

    private void forbidden(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.FORBIDDEN);
    }
}
