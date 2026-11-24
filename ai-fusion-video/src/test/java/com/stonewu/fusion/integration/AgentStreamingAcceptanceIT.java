package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.service.ai.run.AgentExecutionLimits;
import com.stonewu.fusion.service.ai.AgentMessageService;
import com.stonewu.fusion.service.ai.run.stream.AgentRedisStreamStore;
import com.stonewu.fusion.service.ai.run.stream.AgentStreamReader;
import com.stonewu.fusion.service.ai.run.stream.AgentStreamRetentionInitializer;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.data.redis.core.StringRedisTemplate;
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
import java.util.LinkedHashMap;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

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
    @Autowired AgentMessageMapper messages;
    @Autowired AgentRunMapper runs;
    @Autowired AgentEventMapper events;
    @Autowired AgentRedisStreamStore streams;
    @Autowired AgentStreamReader streamReader;
    @Autowired AgentStreamRetentionInitializer retention;
    @Autowired ProjectMapper projects;
    @Autowired AgentMessageService messageService;
    @Autowired StringRedisTemplate redis;
    @LocalServerPort int port;
    private String token;

    @Test void firstChunkReconnectChildAgentAndHistoryRemainConsistent() throws Exception {
        try (var provider = new AgentStreamingProvider(json); var http = HttpClient.newHttpClient()) {
            initialize(provider.url());
            provider.chunks = 240;
            provider.chunkDelayMs = 5;
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
            assertThat(tail.getFirst().path("outputType").asText()).isEqualTo("HISTORY");
            assertThat(tail.subList(1, tail.size())).allSatisfy(event -> assertThat(event.path("sequence").asLong()).isGreaterThan(sequence));
            assertThat(tail.stream().filter(event -> !"replace".equals(event.path("contentMode").asText()))
                    .filter(event -> "CONTENT".equals(event.path("outputType").asText()))
                    .map(event -> event.path("content").asText()).toList()).doesNotContain("首段已到达。\n\n");
            assertThat(text(tail)).contains("首段已到达", "第1段", "第240段");
            assertThat(tail).allSatisfy(event -> assertThat(event.path("schemaVersion").asInt()).isEqualTo(2));
            long[] delays = tail.stream().filter(event -> "CONTENT".equals(event.path("outputType").asText()))
                    .mapToLong(event -> Math.max(0, event.path("_receivedAt").asLong()
                            - Instant.parse(event.path("createdAt").asText()).toEpochMilli())).sorted().toArray();
            long p95 = delays[(int) Math.ceil(delays.length * 0.95) - 1];
            assertThat(p95).as("连续输出事件到达延迟 P95").isLessThan(500);
            assertThat(provider.requests).hasValue(1);
            List<JsonNode> refreshed = readAll(http, "/api/ai/pipeline/reconnect?runId=" + run + "&afterSequence=0");
            assertThat(text(refreshed)).isEqualTo(text(tail));
            assertThat(refreshed).singleElement().satisfies(event -> {
                assertThat(event.path("outputType").asText()).isEqualTo("HISTORY");
                assertThat(event.path("finished").asBoolean()).isTrue();
                assertThat(event.path("historyMessages")).hasSize(1);
            });
            assertCompleteMessages(run, text(refreshed), 1);
            assertThat(streamReader.read(runs.selectByRunId(run), refreshed.getLast().path("sequence").asLong())
                    .blockLast(Duration.ofSeconds(2)).getOutputType()).isEqualTo("HISTORY");
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
                assertThat(event.path("content").asText()).contains("第240段");
            });
            String parentRun = delegated.getFirst().path("runId").asText();
            List<AgentRun> children = runs.selectList(new LambdaQueryWrapper<AgentRun>().eq(AgentRun::getParentRunId, parentRun));
            assertThat(children).singleElement().satisfies(child -> {
                assertThat(child.getStreamRunId()).isEqualTo(parentRun);
                assertThat(child.getStreamProtocolVersion()).isEqualTo(2);
                assertCompleteMessages(child.getRunId(), text(delegated.stream()
                        .filter(event -> child.getRunId().equals(event.path("sourceRunId").asText())).toList()), 1);
            });
            assertCompleteMessages(parentRun, null, 2);
            List<AgentMessage> history = messageService.listByConversation(childConversation);
            assertThat(history).filteredOn(message -> "user".equals(message.getRole())).hasSize(1);
            assertThat(history).anySatisfy(message -> {
                assertThat(message.getParentToolCallId()).isEqualTo("delegate-1");
                assertThat(message.getRole()).isEqualTo("assistant");
                assertThat(message.getContent()).contains("第240段");
            });
            assertThat(messages.selectList(new LambdaQueryWrapper<AgentMessage>()
                    .eq(AgentMessage::getConversationId, childConversation).eq(AgentMessage::getRole, "user")))
                    .as("子任务输入仍保留在数据库中，供执行恢复使用").hasSize(2);
            List<AgentMessage> toolMessages = messages.selectList(new LambdaQueryWrapper<AgentMessage>()
                    .eq(AgentMessage::getRunId, parentRun).eq(AgentMessage::getRole, "tool").orderByAsc(AgentMessage::getMessageOrder));
            assertThat(toolMessages).hasSize(2);
            assertThat(toolMessages.getFirst().getContent()).contains("730001");
            assertThat(toolMessages.getLast().getToolStatus()).isEqualTo("success");
            assertThat(toolMessages.getLast().getContent()).contains("第240段");
            assertThat(provider.requests).hasValue(4);
            Files.createDirectories(STORAGE);
            Files.writeString(STORAGE.resolve("results.json"), json.writeValueAsString(Map.of(
                    "agentscope", "2.0.4", "firstChunkLatencyMs", latency, "eventLatencyP95Ms", p95, "replayedEvents", refreshed.size(),
                    "childEvents", delegated.size(), "providerRequests", provider.requests.get())));
            approval(http, provider, true);
            approval(http, provider, false);
            cancelPartialReply(http, provider);
            lifecycleAndExpiry(http, provider, "ai_media");
            lifecycleAndExpiry(http, provider, "script_full_parse");
            cancelOneChild(http, provider);
            cancelParentAfterTwoChildrenCompleted(http, provider);
            if (Boolean.getBoolean("fusion.streaming.preview")) {
                Path preview = Path.of("target", "agent-streaming-preview.json");
                Files.writeString(preview, json.writeValueAsString(Map.of("backendPort", port, "username", "streaming-test",
                        "password", "Streaming-Test-2026!", "modelId", ID, "projectId", ID, "results", STORAGE.toString())));
                Path stop = Path.of("target", "agent-streaming-preview.stop");
                while (!Files.exists(stop)) Thread.sleep(1000);
            }
        }
    }

    @Test
    void concurrentChildrenShareOneOrderedStreamAndReconnectAcrossPagesWithoutDuplicates() throws Exception {
        String runId = UUID.randomUUID().toString();
        try (var workers = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<?>> writers = new ArrayList<>();
            for (int child = 0; child < 4; child++) {
                String source = "child-" + child;
                writers.add(workers.submit(() -> {
                    for (int index = 0; index < 100; index++) {
                        String identity = source + ':' + index;
                        var value = new AiChatStreamRespVO().setSourceRunId(source).setContent(identity).setOutputType("CONTENT");
                        long first = streams.append(runId, identity, value).getSequence();
                        assertThat(streams.append(runId, identity, value).getSequence()).isEqualTo(first);
                    }
                }));
            }
            for (Future<?> writer : writers) {
                writer.get(15, TimeUnit.SECONDS);
            }
        }
        List<AiChatStreamRespVO> first = streams.readAfter(runId, 0, 256);
        List<AiChatStreamRespVO> tail = streams.readAfter(runId, first.getLast().getSequence(), 256);
        assertThat(first).hasSize(256);
        assertThat(tail).hasSize(144);
        List<AiChatStreamRespVO> all = new ArrayList<>(first);
        all.addAll(tail);
        assertThat(all).extracting(AiChatStreamRespVO::getContent).doesNotHaveDuplicates();
        for (int index = 0; index < all.size(); index++) {
            assertThat(all.get(index).getSequence()).isEqualTo(index + 1L);
        }
        assertThat(streams.lastSequence(runId)).isEqualTo(400);
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
        return start(http, conversation, type, "FULL_ACCESS");
    }

    private HttpResponse<InputStream> start(HttpClient http, String conversation, String type, String mode) throws Exception {
        String body = json.writeValueAsString(Map.of("conversationId", conversation, "message", "验证连续输出", "modelId", ID,
                "projectId", ID, "agentType", type, "category", "assistant", "enableTools", true, "toolExecutionMode", mode,
                "context", Map.of("scriptId", ID)));
        return http.send(request("/api/ai/pipeline/run").header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofInputStream());
    }

    private HttpRequest.Builder request(String path) {
        return HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + path)).header("Authorization", "Bearer " + token);
    }

    private void approval(HttpClient http, AgentStreamingProvider provider, boolean approved) throws Exception {
        provider.gate = false;
        provider.chunks = 2;
        provider.toolName = "save_project";
        provider.toolArguments = json.writeValueAsString(Map.of("projectId", ID,
                "expectedVersion", projects.selectById(ID).getVersion(), "name", "审批后的测试项目"));
        try (var stream = start(http, UUID.randomUUID().toString(), "ai_media", "DEFAULT").body();
             var reader = reader(stream)) {
            JsonNode pending = CompletableFuture.supplyAsync(() -> until(reader, "USER_CONFIRMATION_REQUIRED"))
                    .get(20, TimeUnit.SECONDS);
            String runId = pending.path("runId").asText();
            AiChatStreamRespVO waiting = streamReader.read(runs.selectByRunId(runId), 0).next().block(Duration.ofSeconds(5));
            assertThat(waiting.getRunStatus()).isEqualTo("WAITING_CONFIRMATION");
            assertThat(waiting.getPendingEvents()).anySatisfy(event -> {
                assertThat(event.getOutputType()).isEqualTo("USER_CONFIRMATION_REQUIRED");
                assertThat(event.getReplyId()).isEqualTo(pending.path("replyId").asText());
            });
            assertThat(redis.getExpire(redisKeys(runId).getFirst())).isEqualTo(-1);
            List<AgentMessage> before = toolMessages(runId);
            assertThat(before).singleElement().satisfies(call -> {
                assertThat(call.getToolStatus()).isEqualTo("running");
                assertThat(call.getContent()).isEqualTo(provider.toolArguments);
            });
            String body = json.writeValueAsString(Map.of("runId", runId, "replyId", pending.path("replyId").asText(),
                    "decisions", List.of(Map.of("toolCallId", "write-1", "approved", approved))));
            var response = http.send(request("/api/ai/pipeline/confirm").header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
            assertThat(response.statusCode()).isEqualTo(200);
            assertThat(json.readTree(response.body()).path("data").asBoolean()).isTrue();
            List<JsonNode> tail = CompletableFuture.supplyAsync(() -> throughTerminal(reader)).get(20, TimeUnit.SECONDS);
            assertThat(tail.getFirst().path("outputType").asText()).isEqualTo("USER_CONFIRM_RESULT");
            assertThat(tail.getLast().path("outputType").asText()).isEqualTo("DONE");
            List<AgentMessage> after = toolMessages(runId);
            assertThat(after).hasSize(2);
            assertThat(after.getLast().getToolStatus()).isEqualTo(approved ? "success" : "rejected");
            if (approved) {
                assertThat(after.getLast().getContent()).contains("\"status\":\"success\"");
            }
        }
        provider.toolName = null;
    }

    private void cancelPartialReply(HttpClient http, AgentStreamingProvider provider) throws Exception {
        provider.gate = true;
        provider.release = new CountDownLatch(1);
        try (var stream = start(http, UUID.randomUUID().toString(), "ai_media").body();
             var reader = reader(stream)) {
            JsonNode first = CompletableFuture.supplyAsync(() -> firstContent(reader)).get(12, TimeUnit.SECONDS);
            String runId = first.path("runId").asText();
            var response = http.send(request("/api/ai/pipeline/cancel?runId=" + runId)
                    .POST(HttpRequest.BodyPublishers.noBody()).build(), HttpResponse.BodyHandlers.ofString());
            assertThat(response.statusCode()).isEqualTo(200);
            List<JsonNode> tail = CompletableFuture.supplyAsync(() -> throughTerminal(reader)).get(12, TimeUnit.SECONDS);
            assertThat(tail.getLast().path("outputType").asText()).isEqualTo("CANCELLED");
            assertCompleteMessages(runId, first.path("content").asText(), 1);
            provider.release.countDown();
        }
    }

    private void lifecycleAndExpiry(HttpClient http, AgentStreamingProvider provider, String type) throws Exception {
        provider.gate = false;
        provider.children = 3;
        provider.chunks = 100;
        provider.chunkDelayMs = 20;
        String conversation = UUID.randomUUID().toString();
        String runId;
        try (var opened = start(http, conversation, type).body(); var reader = reader(opened)) {
            JsonNode content = firstScopedContent(reader, "script_full_parse".equals(type));
            runId = content.path("runId").asText();
            assertThat(runs.selectByRunId(runId).getStatus()).isEqualTo("RUNNING");
            assertThat(redis.getExpire(redisKeys(runId).getFirst())).isEqualTo(-1);
        }
        List<JsonNode> reopened = readAll(http, "/api/ai/pipeline/reconnect?runId=" + runId + "&afterSequence=0");
        assertThat(reopened.getFirst().path("outputType").asText()).isEqualTo("HISTORY");
        assertThat(reopened.getFirst().path("pendingEvents")).isNotEmpty();
        assertThat(reopened.getLast().path("outputType").asText()).isEqualTo("DONE");
        assertThat(runs.selectByRunId(runId).getCancelRequestedAt()).isNull();
        if ("script_full_parse".equals(type)) {
            assertThat(children(runId)).hasSize(3).allSatisfy(child -> {
                assertThat(child.getStatus()).isEqualTo("COMPLETED");
                assertThat(child.getCancelRequestedAt()).isNull();
            });
        }
        for (String key : redisKeys(runId)) {
            assertThat(redis.getExpire(key)).as("整轮结束后流和配套索引保留一天").isBetween(86370L, 86400L);
            redis.persist(key);
        }
        retention.initialize().block(Duration.ofSeconds(5));
        for (String key : redisKeys(runId)) {
            assertThat(redis.getExpire(key)).as("启动时也为旧缓存补齐剩余保留期限").isBetween(86370L, 86400L);
        }
        JsonNode completed = readAll(http, "/api/ai/pipeline/reconnect?runId=" + runId).getFirst();
        List<AgentMessage> saved = messageService.listByConversation(conversation);
        long cursor = streams.lastSequence(runId);
        streams.expireAt(runId, Instant.now().minusSeconds(1));
        assertThat(redis.countExistingKeys(redisKeys(runId))).isZero();
        JsonNode expired = readAll(http, "/api/ai/pipeline/reconnect?runId=" + runId + "&afterSequence=" + cursor).getFirst();
        assertThat(expired.path("historyMessages")).isEqualTo(completed.path("historyMessages"));
        assertThat(expired.path("runStatus").asText()).isEqualTo("COMPLETED");
        assertThat(expired.path("sequence").asLong()).isZero();
        var status = http.send(request("/api/ai/pipeline/status?runId=" + runId).GET().build(), HttpResponse.BodyHandlers.ofString());
        assertThat(json.readTree(status.body()).path("data").path("status").asText()).isEqualTo("COMPLETED");
        assertThat(redis.countExistingKeys(redisKeys(runId))).as("状态与历史读取不重新创建已过期流").isZero();
        provider.chunks = 3;
        List<JsonNode> continued;
        try (var next = start(http, conversation, type).body()) {
            continued = readAll(next);
        }
        String nextRun = continued.getFirst().path("runId").asText();
        assertThat(nextRun).isNotEqualTo(runId);
        assertThat(continued.getLast().path("outputType").asText()).isEqualTo("DONE");
        if ("script_full_parse".equals(type)) {
            assertThat(children(nextRun)).hasSize(3).allSatisfy(child -> assertThat(child.getStatus()).isEqualTo("COMPLETED"));
        }
        List<AgentMessage> extended = messageService.listByConversation(conversation);
        assertThat(extended).filteredOn(message -> "user".equals(message.getRole())).hasSize(2);
        assertThat(extended.subList(0, saved.size())).isEqualTo(saved);
        assertThat(extended).extracting(AgentMessage::getMessageOrder).isSorted().doesNotHaveDuplicates();
        assertThat(redis.countExistingKeys(redisKeys(runId))).isZero();

        provider.chunks = 200;
        String cancelledConversation = UUID.randomUUID().toString();
        String cancelledRun;
        try (var opened = start(http, cancelledConversation, type).body(); var reader = reader(opened)) {
            cancelledRun = firstScopedContent(reader, "script_full_parse".equals(type)).path("runId").asText();
        }
        cancel(http, cancelledRun);
        awaitTerminal(cancelledRun);
        for (AgentRun child : children(cancelledRun)) {
            awaitTerminal(child.getRunId());
        }
        long retentionDeadline = System.nanoTime() + Duration.ofSeconds(5).toNanos();
        while (redis.getExpire(redisKeys(cancelledRun).getFirst()) == -1 && System.nanoTime() < retentionDeadline) {
            Thread.sleep(20);
        }
        assertThat(redis.getExpire(redisKeys(cancelledRun).getFirst()))
                .as("没有任何前端订阅时，最后一个子任务提交取消消息后也会设置共享流期限")
                .isBetween(86370L, 86400L);
        JsonNode cancelled = readAll(http, "/api/ai/pipeline/reconnect?runId=" + cancelledRun).getFirst();
        assertThat(cancelled.path("runStatus").asText()).isEqualTo("CANCELLED");
        assertThat(cancelled.path("historyMessages")).isNotEmpty();
        if ("script_full_parse".equals(type)) {
            assertThat(children(cancelledRun)).hasSize(3).allSatisfy(child -> assertThat(child.getStatus()).isEqualTo("CANCELLED"));
        }
        assertThat(messageService.listByConversation(cancelledConversation)).noneSatisfy(message ->
                assertThat(message.getToolStatus()).isEqualTo("running"));
        assertThat(redis.getExpire(redisKeys(cancelledRun).getFirst())).isBetween(86370L, 86400L);
    }

    private void cancelOneChild(HttpClient http, AgentStreamingProvider provider) throws Exception {
        provider.gate = false;
        provider.children = 3;
        provider.chunks = 120;
        provider.chunkDelayMs = 20;
        String runId;
        try (var opened = start(http, UUID.randomUUID().toString(), "script_full_parse").body(); var reader = reader(opened)) {
            runId = firstScopedContent(reader, true).path("runId").asText();
        }
        List<AgentRun> children = children(runId);
        assertThat(children).hasSize(3);
        String childId = children.getFirst().getRunId();
        cancel(http, childId);
        awaitTerminal(childId);
        assertThat(runs.selectByRunId(runId).getStatus()).isEqualTo("RUNNING");
        assertThat(redis.getExpire(redisKeys(runId).getFirst())).as("单个子任务结束不让共享流提前过期").isEqualTo(-1);
        JsonNode childHistory = readAll(http, "/api/ai/pipeline/reconnect?runId=" + childId).getFirst();
        assertThat(childHistory.path("runStatus").asText()).isEqualTo("CANCELLED");
        assertThat(childHistory.path("historyMessages")).allSatisfy(message ->
                assertThat(message.path("parentToolCallId").isNull() || message.path("parentToolCallId").isMissingNode()).isTrue());
        readAll(http, "/api/ai/pipeline/reconnect?runId=" + runId);
        assertThat(children(runId)).filteredOn(child -> !childId.equals(child.getRunId()))
                .allSatisfy(child -> assertThat(child.getStatus()).isEqualTo("COMPLETED"));
        assertThat(runs.selectByRunId(childId).getStatus()).isEqualTo("CANCELLED");
        assertThat(toolMessages(runId)).filteredOn(message -> children.getFirst().getParentToolCallId().equals(message.getToolCallId()))
                .anySatisfy(message -> assertThat(message.getToolStatus()).isEqualTo("cancelled"));
    }

    private List<AgentRun> children(String runId) {
        return runs.selectList(new LambdaQueryWrapper<AgentRun>().eq(AgentRun::getParentRunId, runId).orderByAsc(AgentRun::getId));
    }

    private void cancelParentAfterTwoChildrenCompleted(HttpClient http, AgentStreamingProvider provider) throws Exception {
        provider.chunks = 2;
        provider.chunkDelayMs = 10;
        provider.holdLastChild = true;
        provider.childRequests.set(0);
        String conversation = UUID.randomUUID().toString();
        String runId;
        try (var opened = start(http, conversation, "script_full_parse").body(); var reader = reader(opened)) {
            runId = firstScopedContent(reader, true).path("runId").asText();
        }
        long deadline = System.nanoTime() + Duration.ofSeconds(10).toNanos();
        while (children(runId).stream().filter(child -> "COMPLETED".equals(child.getStatus())).count() < 2
                && System.nanoTime() < deadline) {
            Thread.sleep(20);
        }
        assertThat(children(runId)).filteredOn(child -> "COMPLETED".equals(child.getStatus())).hasSize(2);
        cancel(http, runId);
        awaitTerminal(runId);
        for (AgentRun child : children(runId)) {
            awaitTerminal(child.getRunId());
        }
        deadline = System.nanoTime() + Duration.ofSeconds(5).toNanos();
        while (redis.getExpire(redisKeys(runId).getFirst()) == -1 && System.nanoTime() < deadline) {
            Thread.sleep(20);
        }
        assertThat(redis.getExpire(redisKeys(runId).getFirst())).isBetween(86370L, 86400L);
        var childStatuses = messageService.listByConversation(conversation).stream()
                .filter(message -> message.getChildRunStatus() != null)
                .collect(Collectors.toMap(AgentMessage::getToolCallId, AgentMessage::getChildRunStatus, (left, right) -> right));
        assertThat(childStatuses.values()).containsExactlyInAnyOrder("COMPLETED", "COMPLETED", "CANCELLED");
        provider.lastChildRelease.countDown();
        provider.holdLastChild = false;
    }

    private List<String> redisKeys(String runId) {
        String key = "fv:ai:run:{" + runId + "}:events";
        return List.of(key, key + ":sequence", key + ":identities", key + ":controls");
    }

    private void cancel(HttpClient http, String runId) throws Exception {
        var response = http.send(request("/api/ai/pipeline/cancel?runId=" + runId)
                .POST(HttpRequest.BodyPublishers.noBody()).build(), HttpResponse.BodyHandlers.ofString());
        assertThat(json.readTree(response.body()).path("data").asBoolean()).isTrue();
    }

    private void awaitTerminal(String runId) throws InterruptedException {
        long deadline = System.nanoTime() + Duration.ofSeconds(15).toNanos();
        while (runs.selectByRunId(runId).getProjectionCompletedAt() == null && System.nanoTime() < deadline) {
            Thread.sleep(50);
        }
        assertThat(runs.selectByRunId(runId).getProjectionCompletedAt()).isNotNull();
    }

    private JsonNode firstScopedContent(BufferedReader reader, boolean child) throws Exception {
        return CompletableFuture.supplyAsync(() -> {
            try {
                JsonNode value;
                while ((value = event(reader)) != null) {
                    for (JsonNode expanded : expanded(List.of(value))) {
                        if ("CONTENT".equals(expanded.path("outputType").asText())
                                && (!child || !expanded.path("parentToolCallId").asText().isEmpty())) {
                            return expanded;
                        }
                    }
                }
                throw new AssertionError("未读到指定任务正文");
            } catch (Exception error) {
                throw new IllegalStateException(error);
            }
        }).get(15, TimeUnit.SECONDS);
    }

    private List<AgentMessage> toolMessages(String runId) {
        return messages.selectList(new LambdaQueryWrapper<AgentMessage>().eq(AgentMessage::getRunId, runId)
                .eq(AgentMessage::getRole, "tool").orderByAsc(AgentMessage::getMessageOrder));
    }

    private JsonNode until(BufferedReader reader, String type) {
        try {
            JsonNode value;
            while ((value = event(reader)) != null) {
                for (JsonNode expanded : expanded(List.of(value))) {
                    assertThat(expanded.path("outputType").asText()).isNotEqualTo("ERROR");
                    if (type.equals(expanded.path("outputType").asText())) {
                        return expanded;
                    }
                }
            }
            throw new AssertionError("消息流未出现 " + type);
        } catch (Exception failure) {
            throw new IllegalStateException(failure);
        }
    }

    private List<JsonNode> throughTerminal(BufferedReader reader) {
        try {
            List<JsonNode> values = new ArrayList<>();
            JsonNode value;
            while ((value = event(reader)) != null) {
                values.add(value);
                if (List.of("DONE", "ERROR", "CANCELLED").contains(value.path("outputType").asText())) {
                    return values;
                }
            }
            throw new AssertionError("消息流未完成");
        } catch (Exception failure) {
            throw new IllegalStateException(failure);
        }
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
                for (JsonNode expanded : expanded(List.of(value))) {
                    if ("CONTENT".equals(expanded.path("outputType").asText())) {
                        return expanded;
                    }
                    if ("ERROR".equals(expanded.path("outputType").asText())) {
                        throw new AssertionError(expanded.toString());
                    }
                }
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
        Map<String, String> replies = new LinkedHashMap<>();
        for (JsonNode event : expanded(events)) {
            if (!"CONTENT".equals(event.path("outputType").asText())) {
                continue;
            }
            String id = event.path("messageId").asText();
            String value = event.path("content").asText();
            if ("replace".equals(event.path("contentMode").asText())) {
                replies.put(id, value);
            } else {
                replies.merge(id, value, String::concat);
            }
        }
        return String.join("", replies.values());
    }

    private List<JsonNode> expanded(List<JsonNode> events) {
        List<JsonNode> result = new ArrayList<>();
        for (JsonNode event : events) {
            if (!"HISTORY".equals(event.path("outputType").asText())) {
                result.add(event);
                continue;
            }
            for (JsonNode message : event.path("historyMessages")) {
                if ("assistant".equals(message.path("role").asText())) {
                    ObjectNode content = json.createObjectNode().put("schemaVersion", 2).put("outputType", "CONTENT")
                            .put("runId", event.path("runId").asText()).put("sequence", event.path("sequence").asLong())
                            .put("sourceRunId", message.path("runId").asText()).put("messageId", message.path("projectionKey").asText())
                            .put("contentMode", "replace").put("content", message.path("content").asText())
                            .put("parentToolCallId", message.path("parentToolCallId").asText(""));
                    result.add(content);
                }
            }
            event.path("pendingEvents").forEach(result::add);
        }
        return result;
    }

    private void assertCompleteMessages(String runId, String expectedContent, int count) {
        List<AgentMessage> replies = messages.selectList(new LambdaQueryWrapper<AgentMessage>()
                .eq(AgentMessage::getRunId, runId).eq(AgentMessage::getRole, "assistant"));
        assertThat(replies).hasSize(count);
        if (expectedContent != null) {
            assertThat(replies.getFirst().getContent()).isEqualTo(expectedContent);
        }
        List<AgentEvent> journal = events.selectList(new LambdaQueryWrapper<AgentEvent>().eq(AgentEvent::getRunId, runId));
        assertThat(journal).noneSatisfy(event -> assertThat(event.getRawEventType()).startsWith("PARENT_MIRROR"));
        assertThat(journal).allSatisfy(event -> assertThat(event.getOutputType())
                .isNotIn("CONTENT", "REASONING", "TOOL_CALL_STARTED", "TOOL_CALL", "TOOL_FINISHED"));
        assertThat(journal).hasSizeLessThan(8);
    }
}
