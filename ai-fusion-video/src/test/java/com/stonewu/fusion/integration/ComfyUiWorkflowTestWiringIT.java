package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.assertj.core.api.Assertions.assertThat;
import static org.awaitility.Awaitility.await;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class ComfyUiWorkflowTestWiringIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "comfy-test-wiring-" + UUID.randomUUID()).toAbsolutePath();
    static final ObjectMapper JSON = new ObjectMapper();
    static final AtomicLong IDS = new AtomicLong(39000);
    static final Map<String, Remote> REMOTES = new ConcurrentHashMap<>();
    static final Map<String, Remote> PROMPTS = new ConcurrentHashMap<>();
    static final byte[] PNG = png();
    static final HttpServer PROVIDER = provider();
    static final String WORKFLOW = """
            {"1":{"class_type":"Text","inputs":{"text":"test"}},
             "2":{"class_type":"SaveImage","inputs":{"images":["1",0],"filename_prefix":"trial"}}}
            """;
    static final String INPUTS = """
            {"prompt":[{"nodeId":"1","inputName":"text","valueType":"string"}]}
            """;
    static final String OUTPUTS = """
            [{"nodeId":"2","mediaType":"image","role":"primary"}]
            """;
    @DynamicPropertySource static void properties(DynamicPropertyRegistry p) {
        p.add("spring.datasource.url", MYSQL::getJdbcUrl);
        p.add("spring.datasource.username", MYSQL::getUsername);
        p.add("spring.datasource.password", MYSQL::getPassword);
        p.add("spring.data.redis.host", REDIS::getHost);
        p.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        p.add("spring.data.redis.password", () -> "");
        p.add("app.storage.local-base-path", FILES::toString);
        p.add("app.agent-workspace.local-base-path", () -> FILES.resolve("workspace").toString());
    }
    @Autowired TestRestTemplate http;
    @Autowired TokenService tokens;
    @Autowired JdbcTemplate jdbc;
    @Autowired ComfyUiWorkflowService workflows;
    @AfterAll static void stopProvider() { PROVIDER.stop(0); }

    @Test void submissionReturnsBeforeCompletionAndReplaysOnlyOneProviderPost() {
        var f = fixture();
        var request = f.request();
        var accepted = ok(post("/version/test", request, f));
        String id = accepted.path("id").asText();
        assertThat(accepted.path("status").asText()).isIn("QUEUED", "WAITING", "RUNNING");
        assertThat(ok(post("/version/test", request, f)).path("id").asText()).isEqualTo(id);
        awaitSubmitted(f);
        assertThat(workflows.getVersion(f.version).getTestStatus()).isEqualTo(3);
        f.remote.complete.set(true);
        var done = awaitState(f, id, "SUCCEEDED");
        assertThat(done.path("outputs")).hasSize(1);
        assertThat(done.path("outputs").get(0).path("url").asText()).startsWith("media://");
        assertThat(done.has("request")).isFalse();
        assertThat(done.has("apiConfigHash")).isFalse();
        assertThat(done.path("inputSummary").path("prompt").asText()).isEqualTo(f.prompt);
        assertThat(f.remote.posts.get()).isEqualTo(1);
        assertThat(workflows.getVersion(f.version).getTestStatus()).isEqualTo(1);
        workflows.publishVersion(f.workflow, f.version);
        assertThat(workflows.getVersion(f.version).getPublished()).isTrue();
        var changed = new HashMap<>(request);
        changed.put("inputs", Map.of("prompt", "changed"));
        assertThat(post("/version/test", changed, f).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(ok(get("/version/tests?versionId=" + f.version, f))).hasSize(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_job_event WHERE job_id = ?", Integer.class, id)).isPositive();
    }

    @Test void draftChangesPreventAnOlderSuccessfulRunFromMarkingNewContentPassed() {
        var f = fixture();
        String id = ok(post("/version/test", f.request(), f)).path("id").asText();
        awaitSubmitted(f);
        workflows.updateDraftVersion(f.version, null, WORKFLOW.replace("trial", "changed-trial"), INPUTS, OUTPUTS);
        f.remote.complete.set(true);
        awaitState(f, id, "SUCCEEDED");
        var version = workflows.getVersion(f.version);
        assertThat(version.getWorkflowHash()).isNotEqualTo(f.hash);
        assertThat(version.getTestStatus()).isZero();
        assertThat(version.getLastTestRunId()).isNull();
        assertThat(version.getTestApiConfigHash()).isNull();
    }

    @Test void cancelDoesNotReleaseCapacityUntilTheRemoteConfirmsItsTerminalState() {
        var f = fixture();
        String first = ok(post("/version/test", f.request(), f)).path("id").asText();
        awaitSubmitted(f);
        String second = ok(post("/version/test", f.request(), f)).path("id").asText();
        cancel(f, second);
        awaitState(f, second, "CANCELLED");
        cancel(f, first);
        await().atMost(Duration.ofSeconds(15)).until(() -> f.remote.cancels.get() == 1);
        var cancelling = ok(get("/tests/" + first, f));
        assertThat(cancelling.path("cancelRequested").asBoolean()).isTrue();
        assertThat(cancelling.path("permitHeld").asBoolean()).isTrue();
        assertThat(f.remote.posts.get()).isEqualTo(1);
        f.remote.complete.set(true);
        var done = awaitState(f, first, "CANCELLED");
        assertThat(done.path("permitHeld").asBoolean()).isFalse();
        assertThat(done.path("outputs")).isEmpty();
        assertThat(workflows.getVersion(f.version).getLastTestRunId()).isEqualTo(second);
        assertThat(workflows.getVersion(f.version).getTestStatus()).isEqualTo(4);
    }

    @Test void expiredSubmittedLeaseResumesByQueryWithoutAnotherPost() {
        var f = fixture();
        String id = ok(post("/version/test", f.request(), f)).path("id").asText();
        awaitSubmitted(f);
        awaitState(f, id, "WAITING");
        jdbc.update("""
                UPDATE afv_media_job SET status = 'RUNNING', owner_instance_id = 'crashed-instance',
                    lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?
                """, id);
        f.remote.complete.set(true);
        awaitState(f, id, "SUCCEEDED");
        assertThat(f.remote.posts.get()).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT owner_epoch FROM afv_media_job WHERE id = ?", Long.class, id)).isGreaterThan(1);
    }

    @Test void rejectedSubmissionFailsWithoutPromptPollingOrManualReview() {
        var f = fixture();
        f.remote.rejectSubmission.set(true);
        String id = ok(post("/version/test", f.request(), f)).path("id").asText();
        var failed = awaitState(f, id, "FAILED");
        assertThat(failed.toString()).contains("HTTP 503", "queue is full");
        assertThat(f.remote.posts.get()).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_comfyui_test_run WHERE job_id = ?", Boolean.class, id)).isFalse();
    }

    @Test void unknownPostResponseIsReconciledByThePreassignedPromptId() {
        var f = fixture();
        f.remote.dropReceipt.set(true);
        String id = ok(post("/version/test", f.request(), f)).path("id").asText();
        awaitSubmitted(f);
        f.remote.complete.set(true);
        var done = awaitState(f, id, "SUCCEEDED");
        assertThat(done.path("acknowledged").asBoolean()).isFalse();
        assertThat(f.remote.posts.get()).isEqualTo(1);
    }

    @Test void unacceptedWithdrawalRejectsTheLateHttpRequest() {
        var f = fixture();
        var request = f.request();
        ok(post("/test-requests/" + request.get("requestId") + "/withdraw", Map.of("scope", Map.of("userId", f.user)), f));
        assertThat(post("/version/test", request, f).getStatusCode()).isEqualTo(HttpStatus.GONE);
        assertThat(f.remote.posts.get()).isZero();
    }

    @Test void explicitRecheckContinuesTheOriginalRemoteTaskAndKeepsAnAuditEvent() {
        var f = fixture();
        f.remote.queryFailure.set(true);
        String id = ok(post("/version/test", f.request(), f)).path("id").asText();
        awaitSubmitted(f);
        // 供应商收到 POST 时，工作者可能尚未提交等待阶段；此时注入的错误计数会被正常受理清零。
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT phase = 'POLLING' AND status = 'WAITING' FROM afv_media_job WHERE id = ?", Boolean.class, id)).isTrue());
        jdbc.update("UPDATE afv_comfyui_test_run SET consecutive_errors = 7 WHERE job_id = ?", id);
        jdbc.update("UPDATE afv_media_job SET next_run_at = UTC_TIMESTAMP(6) WHERE id = ?", id);
        var review = awaitState(f, id, "NEEDS_REVIEW");
        assertThat(review.path("permitHeld").asBoolean()).isTrue();
        assertThat(review.path("availableActions").toString()).contains("RECHECK");
        f.remote.queryFailure.set(false);
        f.remote.complete.set(true);
        ok(post("/tests/" + id + "/review", Map.of("scope", Map.of("userId", f.user), "expectedRevision", review.path("revision").asLong(),
                "action", "RECHECK", "note", "在目标 ComfyUI 确认原任务已完成，继续查询原编号"), f));
        awaitState(f, id, "SUCCEEDED");
        assertThat(f.remote.posts.get()).isEqualTo(1);
        await().atMost(Duration.ofSeconds(5)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM afv_media_job_event WHERE job_id = ? AND event_type = 'COMFY_TEST_REVIEW'", Integer.class, id)).isEqualTo(1));
    }

    @Test void withdrawingAcceptedRequestCancelsItsOriginalJobAndCannotCreateAnother() {
        var f = fixture();
        var request = f.request();
        String id = ok(post("/version/test", request, f)).path("id").asText();
        awaitSubmitted(f);
        ok(post("/test-requests/" + request.get("requestId") + "/withdraw", Map.of("scope", Map.of("userId", f.user)), f));
        assertThat(ok(post("/version/test", request, f)).path("id").asText()).isEqualTo(id);
        f.remote.complete.set(true);
        awaitState(f, id, "CANCELLED");
        assertThat(f.remote.posts.get()).isEqualTo(1);
    }

    @Test void scopeAndOwnerAreCheckedForEveryReadAndMutation() {
        var owner = fixture();
        var other = fixture();
        String id = ok(post("/version/test", owner.request(), owner)).path("id").asText();
        assertThat(get("/tests/" + id, other).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var request = new HashMap<>(owner.request());
        request.put("scope", Map.of("userId", owner.user, "teamId", 99999));
        assertThat(post("/version/test", request, owner).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        cancel(owner, id);
        owner.remote.complete.set(true);
        awaitState(owner, id, "CANCELLED");
    }

    @Test void anotherAdminCanCloseARevokedTestOwnerWithoutPublishingOrChangingWorkflowContent() {
        var f = fixture();
        var admin = fixture();
        String id = ok(post("/version/test", f.request(), f)).path("id").asText();
        awaitSubmitted(f);
        String context = jdbc.queryForObject("SELECT context_json FROM afv_media_job WHERE id = ?", String.class, id);
        jdbc.update("UPDATE sys_user_role SET deleted = TRUE WHERE user_id = ?", f.user);
        await().atMost(Duration.ofSeconds(15)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT status FROM afv_media_job WHERE id = ?", String.class, id)).isEqualTo("NEEDS_REVIEW"));
        String path = "/api/system/execution-recovery/COMFY_TEST/" + id;
        var view = ok(http.exchange(path, HttpMethod.GET, new HttpEntity<>(admin.headers), JsonNode.class));
        assertThat(view.path("canResolve").asBoolean()).isTrue();
        assertThat(view.toString()).doesNotContain(f.prompt, "test-key", "workflowHash", "inputSummary");
        f.remote.complete.set(true);
        var request = Map.of("adminUserId", admin.user, "requestId", UUID.randomUUID().toString(), "expectedRevision", view.path("revision").asLong(),
                "expectedFactsHash", view.path("factsHash").asText(),
                "note", "模拟管理员已核实原试运行结束", "providerFinished", true);
        var receipt = ok(http.postForEntity(path + "/confirm-finished", new HttpEntity<>(request, admin.headers), JsonNode.class));
        assertThat(ok(http.postForEntity(path + "/confirm-finished", new HttpEntity<>(request, admin.headers), JsonNode.class))).isEqualTo(receipt);
        assertThat(jdbc.queryForObject("SELECT status FROM afv_media_job WHERE id = ?", String.class, id)).isEqualTo("CANCELLED");
        assertThat(jdbc.queryForObject("SELECT context_json FROM afv_media_job WHERE id = ?", String.class, id)).isEqualTo(context);
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_comfyui_test_run WHERE job_id = ?", Boolean.class, id)).isFalse();
        assertThat(workflows.getVersion(f.version).getTestStatus()).isEqualTo(4);
        assertThat(workflows.getVersion(f.version).getPublished()).isFalse();
        assertThat(jdbc.queryForObject("SELECT workflow_hash FROM afv_comfyui_workflow_version WHERE id = ?", String.class, f.version)).isEqualTo(f.hash);
        assertThat(f.remote.posts).hasValue(1);
    }

    private Fixture fixture() {
        long user = IDS.incrementAndGet();
        String prompt = "comfy-test-" + user;
        jdbc.update("INSERT INTO sys_user(id, username, password) VALUES (?, ?, 'test-only')", user, prompt);
        jdbc.update("INSERT INTO sys_user_role(user_id, role_id) SELECT ?, id FROM sys_role WHERE UPPER(code) = 'ADMIN' AND deleted = FALSE", user);
        if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config", Integer.class) == 0)
            jdbc.update("INSERT INTO afv_storage_config(name, type, base_path, is_default) VALUES ('试运行验收存储', 'local', ?, TRUE)", FILES.toString());
        jdbc.update("INSERT INTO afv_api_config(id, name, platform, image_protocol, api_url, api_key) VALUES (?, ?, 'comfyui', 'comfyui', ?, 'test-key')",
                user, prompt, "http://localhost:" + PROVIDER.getAddress().getPort());
        long workflow = workflows.createWorkflow(user, prompt, prompt, 2, null, 1);
        long version = workflows.createVersion(workflow, null, WORKFLOW, INPUTS, OUTPUTS);
        jdbc.update("UPDATE afv_comfyui_workflow_version SET validation_status = 1 WHERE id = ?", version);
        String hash = jdbc.queryForObject("SELECT workflow_hash FROM afv_comfyui_workflow_version WHERE id = ?", String.class, version);
        var remote = new Remote();
        REMOTES.put(prompt, remote);
        var headers = new HttpHeaders();
        headers.setBearerAuth(tokens.createToken(user, prompt, null).getAccessToken());
        headers.setContentType(MediaType.APPLICATION_JSON);
        return new Fixture(user, workflow, version, hash, prompt, headers, remote);
    }
    private record Fixture(long user, long workflow, long version, String hash, String prompt, HttpHeaders headers, Remote remote) {
        Map<String, Object> request() {
            return Map.of("requestId", UUID.randomUUID().toString(), "scope", Map.of("userId", user), "versionId", version,
                    "workflowHash", hash, "inputs", Map.of("prompt", prompt));
        }
    }
    private void awaitSubmitted(Fixture f) { await().atMost(Duration.ofSeconds(20)).until(() -> f.remote.posts.get() == 1); }
    private void cancel(Fixture f, String id) {
        ok(post("/tests/" + id + "/cancel", Map.of("scope", Map.of("userId", f.user)), f));
        ok(post("/tests/" + id + "/cancel", Map.of("scope", Map.of("userId", f.user)), f));
    }
    private JsonNode awaitState(Fixture f, String id, String state) {
        await().pollInterval(Duration.ofMillis(150)).atMost(Duration.ofSeconds(25)).untilAsserted(() ->
                assertThat(ok(get("/tests/" + id, f)).path("status").asText()).isEqualTo(state));
        return ok(get("/tests/" + id, f));
    }
    private ResponseEntity<JsonNode> get(String suffix, Fixture f) {
        return http.exchange("/api/ai/comfyui/workflow" + suffix, HttpMethod.GET, new HttpEntity<>(f.headers), JsonNode.class);
    }
    private ResponseEntity<JsonNode> post(String suffix, Object body, Fixture f) {
        return http.postForEntity("/api/ai/comfyui/workflow" + suffix, new HttpEntity<>(body, f.headers), JsonNode.class);
    }
    private JsonNode ok(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().path("code").asInt()).as(response.toString()).isZero();
        return response.getBody().path("data");
    }
    private static class Remote {
        final AtomicInteger posts = new AtomicInteger();
        final AtomicInteger cancels = new AtomicInteger();
        final AtomicBoolean complete = new AtomicBoolean();
        final AtomicBoolean dropReceipt = new AtomicBoolean();
        final AtomicBoolean rejectSubmission = new AtomicBoolean();
        final AtomicBoolean queryFailure = new AtomicBoolean();
    }
    private static HttpServer provider() {
        try {
            var server = HttpServer.create(new InetSocketAddress(0), 0);
            server.createContext("/prompt", exchange -> {
                JsonNode body = JSON.readTree(exchange.getRequestBody());
                var remote = REMOTES.get(body.path("prompt").path("1").path("inputs").path("text").asText());
                if (remote == null) { respond(exchange, 400, Map.of("error", "unknown fixture")); return; }
                String id = body.path("prompt_id").asText();
                remote.posts.incrementAndGet();
                if (remote.rejectSubmission.get()) { respond(exchange, 503, Map.of("error", "queue is full")); return; }
                PROMPTS.put(id, remote);
                if (remote.dropReceipt.get()) { exchange.close(); return; }
                respond(exchange, 200, Map.of("prompt_id", id, "number", 1, "node_errors", Map.of()));
            });
            server.createContext("/api/jobs/", exchange -> {
                String path = exchange.getRequestURI().getPath().substring("/api/jobs/".length());
                String id = path.split("/")[0];
                var remote = PROMPTS.get(id);
                if (remote == null) { respond(exchange, 404, Map.of("error", "not found")); return; }
                if (path.endsWith("/cancel")) { remote.cancels.incrementAndGet(); respond(exchange, 200, Map.of("cancelled", true)); return; }
                if (remote.queryFailure.get()) { respond(exchange, 503, Map.of("error", "temporary query outage")); return; }
                respond(exchange, 200, Map.of("id", id, "status", remote.complete.get() ? "completed" : "in_progress",
                        "outputs", remote.complete.get() ? Map.of("2", Map.of("images", List.of(Map.of("filename", "trial.png", "subfolder", "", "type", "output")))) : Map.of()));
            });
            server.createContext("/view", exchange -> {
                exchange.getResponseHeaders().add("Content-Type", "image/png");
                exchange.sendResponseHeaders(200, PNG.length);
                try (var output = exchange.getResponseBody()) { output.write(PNG); }
            });
            server.start();
            return server;
        } catch (IOException error) { throw new IllegalStateException(error); }
    }
    private static void respond(HttpExchange exchange, int status, Object body) throws IOException {
        byte[] bytes = JSON.writeValueAsBytes(body);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, bytes.length);
        try (var output = exchange.getResponseBody()) { output.write(bytes); }
    }
    private static byte[] png() {
        try {
            var bytes = new ByteArrayOutputStream();
            ImageIO.write(new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB), "png", bytes);
            return bytes.toByteArray();
        } catch (IOException error) { throw new IllegalStateException(error); }
    }
}
