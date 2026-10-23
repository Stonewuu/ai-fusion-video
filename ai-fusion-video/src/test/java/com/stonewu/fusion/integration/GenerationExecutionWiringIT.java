package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.system.SystemConfigService;
import com.stonewu.fusion.service.generation.job.GenerationAttemptRepository;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.job.GenerationSubmissionService;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.resource.ResourceToolRequests;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;
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
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.LinkedMultiValueMap;
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
import java.net.URI;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.awaitility.Awaitility.await;

/** 真实 HTTP、调度器、MySQL/Redis 和私有图片登记；供应商为本地受控 HTTP 服务。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class GenerationExecutionWiringIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "generation-execution-wiring-" + UUID.randomUUID()).toAbsolutePath();
    static final AtomicLong IDS = new AtomicLong(28000);
    static final ObjectMapper JSON = new ObjectMapper();
    static final Map<String, Remote> REMOTES = new ConcurrentHashMap<>();
    static final HttpServer PROVIDER = provider();
    static final byte[] PNG = png();
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
    @Autowired SystemConfigService system;
    @Autowired GenerationAttemptRepository attempts;
    @Autowired GenerationSubmissionService submissions;
    @Autowired CanvasJson json;
    @Autowired PlatformTransactionManager transactionManager;
    @AfterAll static void stopProvider() { PROVIDER.stop(0); }

    @Test
    void revokedProjectMemberCannotDeleteACompletedTaskTheyPreviouslySubmitted() {
        var creator = fixture("sync");
        var owner = fixture("sync");
        long project = ok(http.postForEntity("/api/project", new HttpEntity<>(Map.of(
                "requestId", UUID.randomUUID().toString(), "fields", Map.of("name", "撤权后的生成结果")),
                owner.headers), JsonNode.class)).path("id").asLong();
        String membershipPath = "/api/project/" + project + "/members";
        ok(http.postForEntity(membershipPath, new HttpEntity<>(Map.of("requestId", UUID.randomUUID().toString(),
                "expectedRevision", 0, "userId", creator.id, "role", 3), owner.headers), JsonNode.class));
        var request = new HashMap<>(creator.request(1));
        request.put("projectId", project);
        String publicId = submit(creator, request);
        awaitState(creator, publicId, "SUCCEEDED");
        long id = ok(get("/api/generation/image/" + publicId, creator.headers)).path("id").asLong();
        assertThat(ok(get("/api/generation/image/" + id + "/items", owner.headers))).hasSize(1);

        var removal = new HashMap<String, Object>();
        removal.put("requestId", UUID.randomUUID().toString());
        removal.put("expectedRevision", 1);
        removal.put("userId", creator.id);
        removal.put("role", null);
        ok(http.postForEntity(membershipPath, new HttpEntity<>(removal, owner.headers), JsonNode.class));
        assertThat(get("/api/generation/image/" + publicId, creator.headers).getStatusCode())
                .isEqualTo(HttpStatus.FORBIDDEN);
        var deletion = http.exchange("/api/generation/image/" + id, HttpMethod.DELETE,
                new HttpEntity<>(creator.headers), JsonNode.class);
        var ownerRead = get("/api/generation/image/" + publicId, owner.headers);
        assertThat(deletion.getStatusCode()).as("Revoked creator DELETE=%s; owner GET after deletion=%s",
                deletion.getStatusCode(), ownerRead.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        ok(ownerRead);
        assertThat(ok(get("/api/generation/image/" + id + "/items", owner.headers))).hasSize(1);
        ok(http.postForEntity(membershipPath, new HttpEntity<>(Map.of("requestId", UUID.randomUUID().toString(),
                "expectedRevision", 2, "userId", creator.id, "role", 3), owner.headers), JsonNode.class));
        ok(http.exchange("/api/generation/image/" + id, HttpMethod.DELETE,
                new HttpEntity<>(creator.headers), JsonNode.class));
        assertThat(get("/api/generation/image/" + publicId, owner.headers).getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    @Test void scopeMismatchCannotCreateOrWithdrawAnotherScopeRequest() {
        var f = fixture("sync");
        var scope = ok(get("/api/generation/executions/scope", f.headers));
        assertThat(scope.path("userId").asLong()).isEqualTo(f.id);
        var body = new HashMap<>(f.request(1));
        body.put("scope", Map.of("userId", f.id, "teamId", 99999L));
        assertThat(post(f, body).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        var wrong = http.postForEntity("/api/generation/executions/requests/" + body.get("requestId") + "/withdraw",
                new HttpEntity<>(Map.of("kind", "IMAGE", "scope", Map.of("userId", f.id + 1)), f.headers), JsonNode.class);
        assertThat(wrong.getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission WHERE actor_user_id = ?", Integer.class, f.id)).isZero();
    }

    @Test void withdrawingUnknownSubmissionPermanentlyRejectsItsLateArrival() {
        var f = fixture("sync");
        var body = f.request(1);
        var withdrawal = new HttpEntity<>(Map.of("kind", "IMAGE", "scope", Map.of("userId", f.id)), f.headers);
        String path = "/api/generation/executions/requests/" + body.get("requestId") + "/withdraw";
        assertThat(ok(http.postForEntity(path, withdrawal, JsonNode.class)).path("state").asText()).isEqualTo("CANCELLED_BEFORE_ACCEPTANCE");
        assertThat(ok(http.postForEntity(path, withdrawal, JsonNode.class)).path("state").asText()).isEqualTo("CANCELLED_BEFORE_ACCEPTANCE");
        assertThat(post(f, body).getStatusCode()).isEqualTo(HttpStatus.GONE);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission WHERE actor_user_id = ?", Integer.class, f.id)).isZero();
        assertThat(f.remote.calls.get()).isZero();
    }

    @Test void withdrawingAcceptedSubmissionCancelsOriginalAndRetryKeepsItsIdentity() {
        var f = fixture("async");
        var body = f.request(1);
        String taskId = submit(f, body);
        await().atMost(Duration.ofSeconds(20)).untilAsserted(() -> assertThat(f.remote.calls.get()).isEqualTo(1));
        var withdrawal = new HttpEntity<>(Map.of("kind", "IMAGE", "scope", Map.of("userId", f.id)), f.headers);
        var result = ok(http.postForEntity("/api/generation/executions/requests/" + body.get("requestId") + "/withdraw", withdrawal, JsonNode.class));
        assertThat(result.path("taskId").asText()).isEqualTo(taskId);
        assertThat(result.path("state").asText()).isEqualTo("CANCEL_REQUESTED");
        assertThat(submit(f, body)).isEqualTo(taskId);
        f.remote.complete.set(true);
        awaitState(f, taskId, "CANCELLED");
        assertThat(f.remote.calls.get()).isEqualTo(1);
    }

    @Test void nativeBatchAdmissionReplaysOnceAndPublishesOnlyManagedOutputs() {
        var f = fixture("sync");
        var body = f.request(3);
        String taskId = submit(f, body);
        assertThat(submit(f, body)).isEqualTo(taskId);
        var done = awaitState(f, taskId, "SUCCEEDED");
        var taskView = ok(get("/api/generation/image/" + taskId, f.headers));
        assertThat(taskView.path("taskId").asText()).isEqualTo(taskId);
        assertThat(taskView.path("execution").path("id").asText()).isEqualTo(done.path("id").asText());
        assertThat(taskView.path("execution").path("canControl").asBoolean()).isTrue();
        assertThat(taskView.has("task")).isFalse();
        assertThat(taskView.has("executionContextJson")).isFalse();
        assertThat(f.remote.calls.get()).isEqualTo(1);
        assertThat(f.remote.count.get()).isEqualTo(3);
        assertThat(done.path("items")).hasSize(1);
        assertThat(done.path("items").get(0).path("itemIds")).hasSize(3);
        var items = ok(get("/api/generation/image/" + done.path("recordId").asLong() + "/items", f.headers));
        assertThat(items).hasSize(3);
        for (var item : items) {
            assertThat(item.path("imageUrl").asText()).startsWith("media://");
            assertThat(item.path("status").asInt()).isEqualTo(1);
        }
        String media = items.get(0).path("imageUrl").asText().substring("media://".length());
        String url = ok(http.postForEntity("/api/media-resources/" + media + "/access", new HttpEntity<>(f.headers), JsonNode.class)).path("url").asText();
        assertThat(http.getForEntity(url, byte[].class).getBody()).isEqualTo(PNG);
        var changed = new HashMap<>(body);
        changed.put("prompt", "不同的内容");
        assertThat(post(f, changed).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission WHERE actor_user_id = ?", Integer.class, f.id)).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_remote_generation_job WHERE actor_user_id = ? AND permit_held = TRUE", Integer.class, f.id)).isZero();
    }

    @Test void missingBatchOutputsArePartialFailureWithEveryActualResultPreserved() {
        var f = fixture("partial");
        String task = submit(f, f.request(3));
        var done = awaitState(f, task, "PARTIAL_FAILED");
        var items = ok(get("/api/generation/image/" + done.path("recordId").asLong() + "/items", f.headers));
        assertThat(items).hasSize(3);
        assertThat(items.get(0).path("status").asInt()).isEqualTo(1);
        assertThat(items.get(1).path("status").asInt()).isEqualTo(1);
        assertThat(items.get(2).path("status").asInt()).isEqualTo(2);
        assertThat(items.get(2).path("imageUrl").isNull()).isTrue();
    }

    @Test void runningCancellationKeepsCapacityUntilRemoteTerminalAndQueuedCancellationNeverSubmits() {
        var f = fixture("async");
        String first = submit(f, f.request(1));
        await().atMost(Duration.ofSeconds(20)).until(() -> f.remote.calls.get() == 1);
        var secondBody = f.request(1);
        String second = submit(f, secondBody);
        ok(http.postForEntity("/api/generation/image/" + second + "/cancel", new HttpEntity<>(f.headers), JsonNode.class));
        awaitState(f, second, "CANCELLED");
        assertThat(f.remote.calls.get()).isEqualTo(1);
        ok(http.postForEntity("/api/generation/image/" + first + "/cancel", new HttpEntity<>(f.headers), JsonNode.class));
        var stopping = awaitState(f, first, "CANCEL_REQUESTED");
        assertThat(stopping.path("items").get(0).path("remoteCapacityHeld").asBoolean()).isTrue();
        f.remote.complete.set(true);
        var cancelled = awaitState(f, first, "CANCELLED");
        assertThat(cancelled.path("items").get(0).path("remoteCapacityHeld").asBoolean()).isFalse();
        assertThat(f.remote.calls.get()).isEqualTo(1);
        var items = ok(get("/api/generation/image/" + cancelled.path("recordId").asLong() + "/items", f.headers));
        assertThat(items.get(0).path("imageUrl").isNull()).isTrue();
    }

    @Test void explicitProviderErrorEndsExecutionAndReleasesCapacityWithoutReview() {
        var f = fixture("rejected");
        String task = submit(f, f.request(1));
        var failed = awaitState(f, task, "FAILED");
        var item = failed.path("items").get(0);
        assertThat(item.path("remoteCapacityHeld").asBoolean()).isFalse();
        assertThat(item.path("availableActions")).isEmpty();
        assertThat(failed.toString()).contains("HTTP 503", "video queue is full");
        assertThat(f.remote.calls.get()).isEqualTo(1);
        assertThat(item.path("attempts").get(0).path("acknowledged").asBoolean()).isTrue();
        f.remote.complete.set(true);
        String retry = submit(f, f.request(1));
        awaitState(f, retry, "SUCCEEDED");
        assertThat(f.remote.calls.get()).isEqualTo(2);
        assertThat(ok(get("/api/generation/executions/" + failed.path("id").asText(), f.headers)).path("state").asText()).isEqualTo("FAILED");
    }

    @Test void ambiguousSubmissionDoesNotRetryAndRequiresVersionedExplicitReview() {
        var f = fixture("unknown");
        String task = submit(f, f.request(1));
        var unknown = awaitState(f, task, "UNKNOWN_SUBMISSION");
        assertThat(f.remote.calls.get()).isEqualTo(1);
        var item = unknown.path("items").get(0);
        assertThat(item.path("remoteCapacityHeld").asBoolean()).isTrue();
        var stale = Map.of("expectedRevision", item.path("revision").asLong() - 1, "action", "CONFIRM_NOT_ACCEPTED_AND_RETRY", "note", "测试供应商已确认未受理");
        String path = "/api/generation/executions/jobs/" + item.path("id").asText() + "/review";
        assertThat(http.postForEntity(path, new HttpEntity<>(stale, f.headers), JsonNode.class).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        f.remote.complete.set(true);
        ok(http.postForEntity(path, new HttpEntity<>(Map.of("expectedRevision", item.path("revision").asLong(),
                "action", "CONFIRM_NOT_ACCEPTED_AND_RETRY", "note", "受控供应商确认第一次请求未受理，明确重新提交"), f.headers), JsonNode.class));
        awaitState(f, task, "SUCCEEDED");
        assertThat(f.remote.calls.get()).isEqualTo(2);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_attempt WHERE job_id = ?", Integer.class, item.path("id").asText())).isEqualTo(2);
    }

    @Test void anotherUserCannotReadOrReviewAnUncertainExecution() {
        var owner = fixture("unknown");
        var other = fixture("sync");
        String task = submit(owner, owner.request(1));
        var unknown = awaitState(owner, task, "UNKNOWN_SUBMISSION");
        assertThat(get("/api/generation/executions/" + unknown.path("id").asText(), other.headers).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var item = unknown.path("items").get(0);
        var denied = http.postForEntity("/api/generation/executions/jobs/" + item.path("id").asText() + "/review", new HttpEntity<>(Map.of(
                "expectedRevision", item.path("revision").asLong(), "action", "CONFIRM_PROVIDER_FINISHED", "note", "无权作出此决定"), other.headers), JsonNode.class);
        assertThat(denied.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(owner.remote.calls.get()).isEqualTo(1);
    }

    @Test void providerInputTicketSurvivesPollOwnershipChangesAndIsRevokedOnCancellation() throws Exception {
        var f = fixture("async");
        system.setValue(SystemConfigService.RESOURCE_BASE_URL_KEY, "https://media.example.test");
        jdbc.update("UPDATE afv_ai_model SET config = ? WHERE id = ?", JSON.writeValueAsString(Map.of(
                "defaultWidth", 64, "defaultHeight", 64, "asyncMode", true, "asyncTaskPollIntervalSeconds", 1,
                "supportReferenceImages", true, "referenceImageInputFormats", List.of("url"))), f.id);
        var multipart = new LinkedMultiValueMap<String, Object>();
        multipart.add("file", new ByteArrayResource(PNG) { @Override public String getFilename() { return "reference.png"; } });
        multipart.add("requestId", UUID.randomUUID().toString());
        var uploadHeaders = new HttpHeaders();
        uploadHeaders.addAll(f.headers);
        uploadHeaders.setContentType(MediaType.MULTIPART_FORM_DATA);
        String resource = ok(http.postForEntity("/api/media-resources/upload", new HttpEntity<>(multipart, uploadHeaders), JsonNode.class)).path("id").asText();
        var body = new HashMap<>(f.request(1));
        body.put("refImageUrls", JSON.writeValueAsString(List.of("media://" + resource)));
        String task = submit(f, body);
        await().atMost(Duration.ofSeconds(20)).until(() -> f.remote.inputUrl.get() != null);
        var published = URI.create(f.remote.inputUrl.get());
        assertThat(published.getHost()).isEqualTo("media.example.test");
        // 在受控 HTTP 测试中将公网网关映射到本地应用，不放宽生产的公网地址校验。
        String inputUrl = http.getRootUri() + published.getRawPath() + "?" + published.getRawQuery();
        assertThat(http.getForEntity(inputUrl, byte[].class).getBody()).isEqualTo(PNG);
        var execution = ok(get("/api/generation/image/" + task + "/execution", f.headers));
        String job = execution.path("items").get(0).path("id").asText();
        await().atMost(Duration.ofSeconds(15)).until(() -> jdbc.queryForObject(
                "SELECT owner_epoch FROM afv_remote_generation_job WHERE id = ?", Long.class, job) >= 3);
        assertThat(http.getForEntity(inputUrl, byte[].class).getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_access_ticket WHERE resource_id = ? AND purpose = 'GENERATION' AND generation_job_id = ? AND generation_attempt_id IS NOT NULL",
                Integer.class, resource, job)).isEqualTo(1);
        ok(http.postForEntity("/api/generation/image/" + task + "/cancel", new HttpEntity<>(f.headers), JsonNode.class));
        assertThat(http.getForEntity(inputUrl, byte[].class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        f.remote.complete.set(true);
        awaitState(f, task, "CANCELLED");
    }

    private Fixture fixture(String mode) {
        long id = IDS.incrementAndGet();
        String prompt = "generation-test-" + id;
        var remote = new Remote(mode, prompt);
        REMOTES.put(prompt, remote);
        jdbc.update("INSERT INTO sys_user(id, username, password) VALUES (?, ?, 'test-only')", id, prompt);
        if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config", Integer.class) == 0) {
            jdbc.update("INSERT INTO afv_storage_config(name, type, base_path, is_default) VALUES ('执行验收存储', 'local', ?, TRUE)", FILES.toString());
        }
        String protocol = mode.equals("async") ? "newapi" : "openai";
        jdbc.update("INSERT INTO afv_api_config(id, name, platform, image_protocol, api_url, api_key) VALUES (?, ?, ?, ?, ?, 'test-key')",
                id, prompt, mode.equals("async") ? "newapi" : "openai_compatible", protocol, "http://localhost:" + PROVIDER.getAddress().getPort());
        String config = mode.equals("async") ? "{\"defaultWidth\":64,\"defaultHeight\":64,\"asyncMode\":true,\"asyncTaskPollIntervalSeconds\":1}" : "{\"defaultWidth\":64,\"defaultHeight\":64}";
        jdbc.update("""
                INSERT INTO afv_ai_model(id, name, code, model_type, model_protocol, api_config_id, config, max_concurrency,
                    multimodal_input_types, multimodal_input_transports, reasoning_effort_levels)
                VALUES (?, ?, ?, 2, ?, ?, ?, 1, JSON_ARRAY(), JSON_OBJECT(), JSON_ARRAY())
                """, id, prompt, "gpt-image-2", protocol, id, config);
        var headers = new HttpHeaders();
        headers.setBearerAuth(tokens.createToken(id, prompt, null).getAccessToken());
        headers.setContentType(MediaType.APPLICATION_JSON);
        return new Fixture(id, prompt, headers, remote);
    }
    @Test void aLateSuccessfulReceiptCanBeRestoredWithoutAnotherProviderPost() {
        var f = fixture("unknown");
        String taskId = submit(f, f.request(1));
        var unknown = awaitState(f, taskId, "UNKNOWN_SUBMISSION");
        var item = unknown.path("items").get(0);
        String jobId = item.path("id").asText();
        String original = jdbc.queryForObject("SELECT active_attempt_id FROM afv_remote_generation_job WHERE id = ?", String.class, jobId);
        new TransactionTemplate(transactionManager).executeWithoutResult(status -> {
            jdbc.queryForList("SELECT id FROM afv_generation_dispatch_lock WHERE id = 1 FOR UPDATE");
            var review = Map.of("expectedRevision", item.path("revision").asLong(), "action", "CONFIRM_NOT_ACCEPTED_AND_RETRY", "note", "先核实未受理，随后收到迟到回执");
            ok(http.postForEntity("/api/generation/executions/jobs/" + jobId + "/review", new HttpEntity<>(review, f.headers), JsonNode.class));
            attempts.acknowledge(original, GenerationProviderReceipt.completed(RemoteGenerationResult.success(List.of(
                    RemoteGenerationResult.Output.image("data:image/png;base64," + Base64.getEncoder().encodeToString(PNG))))));
        });
        var needsReview = awaitState(f, taskId, "NEEDS_REVIEW").path("items").get(0);
        assertThat(needsReview.path("availableActions").toString()).contains("RESTORE_ACCEPTED_RECEIPT");
        ok(http.postForEntity("/api/generation/executions/jobs/" + jobId + "/review", new HttpEntity<>(Map.of(
                "expectedRevision", needsReview.path("revision").asLong(), "action", "RESTORE_ACCEPTED_RECEIPT", "note", "恢复供应商已确认的原请求"), f.headers), JsonNode.class));
        awaitState(f, taskId, "SUCCEEDED");
        assertThat(f.remote.calls.get()).isEqualTo(1);
    }

    @Test void referenceDownloadFailureEndsBeforePaidIntentAndReleasesCapacity() throws Exception {
        var f = fixture("preparation-error");
        String task = submit(f, referenceRequest(f));
        var execution = awaitState(f, task, "FAILED");
        assertThat(f.remote.inputReads.get()).isEqualTo(1);
        assertThat(f.remote.calls.get()).isZero();
        assertUnsubmitted(execution.path("items").get(0).path("id").asText());
    }

    @Test void cancellationDuringReferenceDownloadDoesNotSubmitWhenPreparationReturns() throws Exception {
        var f = fixture("preparation-wait");
        String task = submit(f, referenceRequest(f));
        try {
            assertThat(f.remote.inputStarted.await(20, TimeUnit.SECONDS)).isTrue();
            var execution = ok(get("/api/generation/image/" + task + "/execution", f.headers));
            String jobId = execution.path("items").get(0).path("id").asText();
            assertThat(jdbc.queryForObject("SELECT phase FROM afv_remote_generation_job WHERE id = ?", String.class, jobId))
                    .isEqualTo("PREPARING");
            ok(http.postForEntity("/api/generation/image/" + task + "/cancel", new HttpEntity<>(f.headers), JsonNode.class));
        } finally { f.remote.inputReleased.countDown(); }
        var cancelled = awaitState(f, task, "CANCELLED");
        assertThat(f.remote.calls.get()).isZero();
        assertUnsubmitted(cancelled.path("items").get(0).path("id").asText());
    }

    @Test void preparedMultipartUsesFetchedImageOnceAndPublishesOneManagedResult() throws Exception {
        var f = fixture("preparation-success");
        String task = submit(f, referenceRequest(f));
        var succeeded = awaitState(f, task, "SUCCEEDED");
        assertThat(f.remote.inputReads.get()).isEqualTo(1);
        assertThat(f.remote.calls.get()).isEqualTo(1);
        assertThat(f.remote.contentType.get()).startsWith("multipart/form-data");
        String jobId = succeeded.path("items").get(0).path("id").asText();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_attempt WHERE job_id = ? AND submitted_at IS NOT NULL",
                Integer.class, jobId)).isEqualTo(1);
    }

    @Test void adapterCannotMarkInvalidMultipartInputAsAnUncertainPaidSubmission() throws Exception {
        var f = fixture("preparation-invalid-data");
        var request = referenceRequest(f);
        request.put("refImageUrls", "[\"data:image/png;base64\"]");
        String task = submit(f, request);
        var execution = awaitState(f, task, "FAILED");
        assertUnsubmitted(execution.path("items").get(0).path("id").asText());
        assertThat(f.remote.inputReads.get()).isZero();
        assertThat(f.remote.calls.get()).isZero();
    }

    @Test void longAgentToolIdentityUsesBoundedRequestIdAndReplaysOnePaidSubmission() {
        var f = fixture("sync");
        var actor = agent(f, "call:" + "long-id/".repeat(14));
        String requestId = ResourceToolRequests.requestId(json, actor);
        assertThat(requestId).hasSize(70).matches("agent:[a-f0-9]{64}");
        var first = submissions.image(agentImage(f), requestId, actor, null);
        assertThat(submissions.image(agentImage(f), requestId, actor, null).id()).isEqualTo(first.id());
        awaitState(f, first.publicTaskId(), "SUCCEEDED");
        assertThat(f.remote.calls.get()).isEqualTo(1);
        assertThat(jdbc.queryForList("SELECT tool_call_id FROM afv_media_resource WHERE agent_run_id = ?", String.class, actor.agentRunId()))
                .containsExactly(actor.toolCallId());
        var changed = agentImage(f);
        changed.setPrompt("changed contents");
        assertThatThrownBy(() -> submissions.image(changed, "another-valid-request", actor, null)).hasMessageContaining("不同的生成内容");
        var oversized = new ResourceCommandContext(actor.actor(), "agent", null, actor.agentRunId(), "x".repeat(129),
                actor.ownerInstanceId(), actor.ownerEpoch());
        assertThatThrownBy(() -> submissions.image(agentImage(f), ResourceToolRequests.requestId(json, oversized), oversized, null))
                .hasMessageContaining("最多 128 字符");
        var nonAscii = new ResourceCommandContext(actor.actor(), "agent", null, actor.agentRunId(), "非 ASCII 调用",
                actor.ownerInstanceId(), actor.ownerEpoch());
        assertThatThrownBy(() -> submissions.image(agentImage(f), ResourceToolRequests.requestId(json, nonAscii), nonAscii, null))
                .hasMessageContaining("ASCII");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission WHERE actor_user_id = ?", Integer.class, f.id)).isEqualTo(1);
    }

    @Test void oldAgentRequestKeepsItsReceiptWhenToolRequestEncodingChanges() {
        var f = fixture("sync");
        var actor = agent(f, "legacy-call");
        String original = "tool:" + actor.agentRunId() + ":" + actor.toolCallId();
        var first = submissions.image(agentImage(f), original, actor, null);
        var replay = submissions.image(agentImage(f), ResourceToolRequests.requestId(json, actor), actor, null);
        assertThat(replay.id()).isEqualTo(first.id());
        assertThat(jdbc.queryForObject("SELECT request_id FROM afv_generation_submission WHERE id = ?", String.class, first.id())).isEqualTo(original);
        awaitState(f, first.publicTaskId(), "SUCCEEDED");
        assertThat(f.remote.calls.get()).isEqualTo(1);
    }

    private ResourceCommandContext agent(Fixture f, String call) {
        String run = UUID.randomUUID().toString();
        jdbc.update("""
                INSERT INTO afv_agent_run(run_id, conversation_id, user_id, kernel_fingerprint,
                    agent_definition_snapshot_json, agent_state_session_id, status, owner_instance_id, owner_epoch,
                    lease_until, deadline_at, started_at)
                VALUES (?, ?, ?, 'generation-identity-test', '{}', ?, 'RUNNING', 'identity-test-owner', 1,
                    TIMESTAMPADD(MINUTE, 10, UTC_TIMESTAMP(6)), TIMESTAMPADD(HOUR, 1, UTC_TIMESTAMP(6)), UTC_TIMESTAMP(6))
                """, run, run, f.id, run);
        return new ResourceCommandContext(new CanvasActor(f.id, null), "agent", null, run, call, "identity-test-owner", 1L);
    }

    private ImageTask agentImage(Fixture f) {
        return ImageTask.builder().userId(f.id).modelId(f.id).prompt(f.prompt).count(1).width(64).height(64).build();
    }

    private Map<String, Object> referenceRequest(Fixture f) throws Exception {
        jdbc.update("UPDATE afv_ai_model SET config = ? WHERE id = ?", JSON.writeValueAsString(Map.of(
                "defaultWidth", 64, "defaultHeight", 64, "supportReferenceImages", true,
                "referenceImageInputFormats", List.of("data_uri"))), f.id);
        var request = new HashMap<>(f.request(1));
        request.put("refImageUrls", JSON.writeValueAsString(List.of("http://localhost:" + PROVIDER.getAddress().getPort()
                + "/preparation-input/" + f.prompt)));
        return request;
    }

    private void assertUnsubmitted(String jobId) {
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_attempt WHERE job_id = ? AND submitted_at IS NOT NULL",
                Integer.class, jobId)).isZero();
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_remote_generation_job WHERE id = ?", Boolean.class, jobId)).isFalse();
    }

    private record Fixture(long id, String prompt, HttpHeaders headers, Remote remote) {
        Map<String, Object> request(int count) {
            return Map.of("requestId", UUID.randomUUID().toString(), "scope", Map.of("userId", id), "modelId", id, "prompt", prompt, "width", 64, "height", 64, "count", count);
        }
    }
    private String submit(Fixture f, Map<String, Object> body) { return ok(post(f, body)).asText(); }
    private ResponseEntity<JsonNode> post(Fixture f, Map<String, Object> body) {
        return http.postForEntity("/api/generation/image/submit", new HttpEntity<>(body, f.headers), JsonNode.class);
    }
    private ResponseEntity<JsonNode> get(String path, HttpHeaders headers) {
        return http.exchange(path, HttpMethod.GET, new HttpEntity<>(headers), JsonNode.class);
    }
    private JsonNode awaitState(Fixture f, String task, String state) {
        String path = "/api/generation/image/" + task + "/execution";
        await().pollInterval(Duration.ofMillis(150)).atMost(Duration.ofSeconds(25)).untilAsserted(() -> {
            var execution = ok(get(path, f.headers));
            assertThat(execution.path("state").asText()).as(execution.toString()).isEqualTo(state);
        });
        return ok(get(path, f.headers));
    }
    private JsonNode ok(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().path("code").asInt()).as(response.toString()).isZero();
        return response.getBody().path("data");
    }
    private static final class Remote {
        final String mode;
        final String id;
        final AtomicInteger calls = new AtomicInteger();
        final AtomicInteger count = new AtomicInteger();
        final AtomicBoolean complete = new AtomicBoolean();
        final AtomicReference<String> inputUrl = new AtomicReference<>();
        final AtomicReference<String> contentType = new AtomicReference<>();
        final AtomicInteger inputReads = new AtomicInteger();
        final CountDownLatch inputStarted = new CountDownLatch(1);
        final CountDownLatch inputReleased = new CountDownLatch(1);
        Remote(String mode, String id) { this.mode = mode; this.id = id; }
    }
    private static HttpServer provider() {
        try {
            var server = HttpServer.create(new InetSocketAddress(0), 0);
            server.createContext("/preparation-input/", exchange -> {
                var remote = REMOTES.get(exchange.getRequestURI().getPath().substring("/preparation-input/".length()));
                if (remote == null) { respond(exchange, 404, Map.of("error", "unknown fixture")); return; }
                remote.inputReads.incrementAndGet();
                remote.inputStarted.countDown();
                if (remote.mode.equals("preparation-wait")) {
                    try {
                        if (!remote.inputReleased.await(20, TimeUnit.SECONDS)) throw new IOException("fixture was not released");
                    } catch (InterruptedException failure) { Thread.currentThread().interrupt(); throw new IOException(failure); }
                }
                if (remote.mode.equals("preparation-error")) { respond(exchange, 404, Map.of("error", "source no longer available")); return; }
                exchange.getResponseHeaders().add("Content-Type", "image/png");
                exchange.sendResponseHeaders(200, PNG.length);
                try (var output = exchange.getResponseBody()) { output.write(PNG); }
            });
            server.createContext("/v1/images/edits", exchange -> {
                String request = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.ISO_8859_1);
                var remote = REMOTES.values().stream().filter(value -> request.contains(value.id)).findFirst().orElse(null);
                if (remote == null) { respond(exchange, 400, Map.of("error", "unknown fixture")); return; }
                remote.calls.incrementAndGet();
                remote.contentType.set(exchange.getRequestHeaders().getFirst("Content-Type"));
                respond(exchange, 200, Map.of("data", List.of(Map.of("b64_json", Base64.getEncoder().encodeToString(PNG)))));
            });
            server.createContext("/v1/images/generations", exchange -> {
                JsonNode request = JSON.readTree(exchange.getRequestBody());
                var remote = REMOTES.get(request.path("prompt").asText());
                if (remote == null) { respond(exchange, 400, Map.of("error", "unknown test request")); return; }
                remote.calls.incrementAndGet();
                var refs = request.get("image_urls");
                if (refs != null && refs.isArray() && !refs.isEmpty()) remote.inputUrl.set(refs.get(0).asText());
                int n = request.path("n").asInt(1);
                remote.count.set(n);
                if (remote.mode.equals("unknown") && !remote.complete.get()) { exchange.close(); return; }
                if (remote.mode.equals("rejected") && !remote.complete.get()) { respond(exchange, 503, Map.of("error", Map.of("message", "video queue is full, please retry later"))); return; }
                if (remote.mode.equals("async")) { respond(exchange, 200, Map.of("task_id", remote.id)); return; }
                var output = JSON.createArrayNode();
                for (int i = 0; i < (remote.mode.equals("partial") ? n - 1 : n); i++) output.addObject().put("b64_json", Base64.getEncoder().encodeToString(PNG));
                respond(exchange, 200, Map.of("data", output));
            });
            server.createContext("/v1/tasks/", exchange -> {
                String id = exchange.getRequestURI().getPath().substring("/v1/tasks/".length());
                var remote = REMOTES.get(id);
                if (remote == null) { respond(exchange, 404, Map.of("error", "not found")); return; }
                respond(exchange, 200, Map.of("data", Map.of("id", id, "status", remote.complete.get() ? "completed" : "processing",
                        "result", Map.of("images", List.of(Map.of("url", "data:image/png;base64," + Base64.getEncoder().encodeToString(PNG)))))));
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
