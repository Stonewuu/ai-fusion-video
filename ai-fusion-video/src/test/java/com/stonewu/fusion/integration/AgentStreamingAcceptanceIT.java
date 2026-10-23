package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.ai.run.AgentExecutionLimits;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;

/** 完整 HTTP→真实 AgentScope→本地流式供应商→MySQL/Redis→HTTP 接续验收。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class AgentStreamingAcceptanceIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    private static final Path STORAGE = Path.of("target", "streaming-acceptance-" + UUID.randomUUID()).toAbsolutePath();
    private static final long ID = 730001;

    @DynamicPropertySource static void properties(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url", MYSQL::getJdbcUrl);
        properties.add("spring.datasource.username", MYSQL::getUsername);
        properties.add("spring.datasource.password", MYSQL::getPassword);
        properties.add("spring.data.redis.host", REDIS::getHost);
        properties.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        properties.add("spring.data.redis.password", () -> "");
        properties.add("app.storage.local-base-path", STORAGE::toString);
        properties.add("app.agent-workspace.local-base-path", () -> STORAGE.resolve("workspace").toString());
        properties.add("fusion.agentscope.v2.execution.run-timeout", () -> "0s");
        properties.add("fusion.agentscope.v2.execution.max-iters", () -> "0");
    }

    @Autowired ObjectMapper json;
    @Autowired JdbcTemplate jdbc;
    @Autowired TokenService tokens;
    @Autowired PasswordEncoder passwords;
    @LocalServerPort int port;
    private String token;

    @Test void firstChunkReconnectChildAgentAndHistoryRemainConsistent() throws Exception {
        try (var provider = new AgentStreamingProvider(json); var http = HttpClient.newHttpClient()) {
            initialize(provider.url());
            provider.gate = true;
            String conversation = UUID.randomUUID().toString();
            HttpResponse<InputStream> opened = start(http, conversation, "ai_media");
            assertThat(opened.statusCode()).isEqualTo(200);
            assertThat(opened.headers().firstValue("X-Accel-Buffering")).contains("no");
            JsonNode first;
            long latency;
            try (var reader = reader(opened.body())) {
                first = CompletableFuture.supplyAsync(() -> firstContent(reader)).get(12, TimeUnit.SECONDS);
                latency = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - provider.firstTextNanos);
                assertThat(first.path("content").asText()).isEqualTo("首段已到达。\n\n");
                assertThat(latency).as("供应商首段到 HTTP 消费者的延迟").isLessThan(1500);
                assertThat(provider.release.getCount()).isEqualTo(1);
            }
            provider.release.countDown();
            String run = first.path("runId").asText();
            long sequence = first.path("sequence").asLong();
            List<JsonNode> tail = readAll(http, "/api/ai/pipeline/reconnect?runId=" + run + "&afterSequence=" + sequence);
            assertThat(tail).isNotEmpty();
            assertThat(tail.getLast().path("outputType").asText()).isEqualTo("DONE");
            assertThat(tail).allSatisfy(event -> assertThat(event.path("sequence").asLong()).isGreaterThan(sequence));
            assertThat(text(tail)).doesNotContain("首段已到达").contains("第1段", "第20段");
            long[] delays = tail.stream().filter(event -> "CONTENT".equals(event.path("outputType").asText()))
                    .mapToLong(event -> Math.max(0, event.path("_receivedAt").asLong()
                            - Instant.parse(event.path("createdAt").asText()).toEpochMilli())).sorted().toArray();
            long p95 = delays[(int) Math.ceil(delays.length * 0.95) - 1];
            assertThat(p95).as("连续输出事件到达延迟 P95").isLessThan(500);
            assertThat(provider.requests).hasValue(1);
            List<JsonNode> refreshed = readAll(http, "/api/ai/pipeline/reconnect?runId=" + run + "&afterSequence=0");
            assertThat(text(refreshed)).isEqualTo(first.path("content").asText() + text(tail));
            assertThat(provider.requests).hasValue(1);
            assertThat(jdbc.queryForObject("SELECT deadline_at FROM afv_agent_run WHERE run_id=?", LocalDateTime.class, run).toInstant(ZoneOffset.UTC))
                    .isEqualTo(AgentExecutionLimits.UNLIMITED_DEADLINE);
            Files.createDirectories(STORAGE);
            Files.writeString(STORAGE.resolve("root-results.json"), json.writeValueAsString(Map.of(
                    "firstChunkLatencyMs", latency, "eventLatencyP95Ms", p95, "replayedEvents", refreshed.size(), "providerRequests", provider.requests.get())));

            String childConversation = UUID.randomUUID().toString();
            List<JsonNode> delegated;
            var childResponse = start(http, childConversation, "script_full_parse");
            assertThat(childResponse.statusCode()).isEqualTo(200);
            assertThat(childResponse.headers().firstValue("Content-Type").orElse("")).contains("text/event-stream");
            try (var stream = childResponse.body()) {
                delegated = readAll(stream);
            }
            assertThat(delegated).isNotEmpty();
            assertThat(delegated.getLast().path("outputType").asText()).isEqualTo("DONE");
            assertThat(delegated).anySatisfy(event -> {
                assertThat(event.path("parentToolCallId").asText()).isEqualTo("delegate-1");
                assertThat(event.path("content").asText()).contains("第20段");
            });
            assertThat(provider.requests).hasValue(4);
            Files.createDirectories(STORAGE);
            Files.writeString(STORAGE.resolve("results.json"), json.writeValueAsString(Map.of(
                    "agentscope", "2.0.4", "firstChunkLatencyMs", latency, "eventLatencyP95Ms", p95, "replayedEvents", refreshed.size(),
                    "childEvents", delegated.size(), "providerRequests", provider.requests.get())));
            if (Boolean.getBoolean("fusion.streaming.preview")) {
                Path preview = Path.of("target", "agent-streaming-preview.json");
                Files.writeString(preview, json.writeValueAsString(Map.of("backendPort", port, "username", "streaming-test",
                        "password", "Streaming-Test-2026!", "modelId", ID, "projectId", ID, "results", STORAGE.toString())));
                Path stop = Path.of("target", "agent-streaming-preview.stop");
                while (!Files.exists(stop)) Thread.sleep(1000);
            }
        }
    }

    private void initialize(String url) {
        jdbc.update("INSERT INTO sys_user(id,username,password,nickname) VALUES(?,'streaming-test',?,'流式验收')", ID, passwords.encode("Streaming-Test-2026!"));
        jdbc.update("INSERT INTO sys_user_role(user_id,role_id) SELECT ?,id FROM sys_role WHERE code IN ('admin','user')", ID);
        jdbc.update("INSERT INTO afv_project(id,name,owner_type,owner_id) VALUES(?,'流式验收项目',1,?)", ID, ID);
        jdbc.update("INSERT INTO afv_script(id,project_id,title) VALUES(?,?,'流式验收剧本')", ID, ID);
        jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('流式验收存储','local',?,TRUE)", STORAGE.toString());
        jdbc.update("INSERT INTO afv_api_config(id,name,platform,text_protocol,api_url,api_key) VALUES(?,'本机流式验收','openai_compatible','openai_compatible',?,'test-key')", ID, url);
        jdbc.update("""
                INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels)
                VALUES(?,'本机流式验收','streaming-test',1,'openai_compatible',?,'{}',4,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())
                """, ID, ID);
        token = tokens.createToken(ID, "streaming-test", null).getAccessToken();
    }

    private HttpResponse<InputStream> start(HttpClient http, String conversation, String type) throws Exception {
        String body = json.writeValueAsString(Map.of("conversationId", conversation, "message", "验证连续输出", "modelId", ID,
                "projectId", ID, "agentType", type, "category", "assistant", "enableTools", true, "toolExecutionMode", "FULL_ACCESS",
                "context", Map.of("scriptId", ID)));
        return http.send(request("/api/ai/pipeline/run").header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofInputStream());
    }

    private HttpRequest.Builder request(String path) {
        return HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + path)).header("Authorization", "Bearer " + token);
    }

    private List<JsonNode> readAll(HttpClient http, String path) throws Exception {
        var response = http.send(request(path).GET().build(), HttpResponse.BodyHandlers.ofInputStream());
        assertThat(response.statusCode()).isEqualTo(200);
        try (var stream = response.body()) { return readAll(stream); }
    }

    private List<JsonNode> readAll(InputStream stream) throws Exception {
        return CompletableFuture.supplyAsync(() -> {
            List<JsonNode> values = new ArrayList<>();
            try (var reader = reader(stream)) {
                JsonNode value;
                while ((value = event(reader)) != null) values.add(value);
                return values;
            } catch (Exception failure) { throw new IllegalStateException(failure); }
        }).get(30, TimeUnit.SECONDS);
    }

    private BufferedReader reader(InputStream stream) { return new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8)); }
    private JsonNode firstContent(BufferedReader reader) {
        try {
            JsonNode value;
            while ((value = event(reader)) != null) {
                if ("CONTENT".equals(value.path("outputType").asText())) return value;
                if ("ERROR".equals(value.path("outputType").asText())) throw new AssertionError(value.toString());
            }
            throw new AssertionError("SSE 未返回正文");
        } catch (Exception failure) { throw new IllegalStateException(failure); }
    }
    private JsonNode event(BufferedReader reader) throws Exception {
        String line;
        while ((line = reader.readLine()) != null) {
            if (line.startsWith("data:")) {
                ObjectNode value = (ObjectNode) json.readTree(line.substring(5));
                value.put("_receivedAt", System.currentTimeMillis());
                return value;
            }
        }
        return null;
    }
    private String text(List<JsonNode> events) {
        return events.stream().filter(event -> "CONTENT".equals(event.path("outputType").asText()))
                .map(event -> event.path("content").asText()).reduce("", String::concat);
    }
}
