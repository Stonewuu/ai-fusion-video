package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.mapper.ai.AiModelMapper;
import com.stonewu.fusion.mapper.canvas.CanvasRunMapper;
import com.stonewu.fusion.mapper.canvas.CanvasTextJobMapper;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;
import com.stonewu.fusion.mapper.generation.AsyncOutboxMapper;
import com.stonewu.fusion.mapper.generation.GenerationAttemptMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.job.GenerationAttemptRepository;
import com.stonewu.fusion.service.generation.job.GenerationJobException;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 真实 MySQL 验证受理投递、跨实例领取和外部请求不确定窗口。
 */
@Testcontainers(disabledWithoutDocker = false)
class GenerationJobPersistenceIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("generation_jobs_test").withUsername("afv").withPassword("afv-test")
            .withCommand("--max_allowed_packet=64M", "--sort_buffer_size=256K");

    @Configuration
    @EnableTransactionManagement
    static class Transactions {
    }

    static AnnotationConfigApplicationContext context;
    static JdbcTemplate jdbc;
    static TransactionTemplate transactions;
    static GenerationJobRepository jobs;
    static GenerationAttemptRepository attempts;
    static GenerationOutbox outbox;
    static CanvasJson json;

    @Test
    void outboxBackoffPreservesAggregateOrderWithoutBlockingOtherRuns() {
        String broken = UUID.randomUUID().toString(), independent = UUID.randomUUID().toString();
        jdbc.execute(
                "ALTER TABLE afv_generation_event ADD CONSTRAINT ck_test_outbox_failure CHECK (submission_id <> '" + broken + "')");
        try {
            transactions.executeWithoutResult(status -> {
                outbox.enqueue("failure:1", "GENERATION_CHANGED", broken, GenerationOutbox.state(broken, 1, "RUNNING"));
                outbox.enqueue("failure:2", "GENERATION_CHANGED", broken,
                        GenerationOutbox.state(broken, 2, "SUCCEEDED"));
                outbox.enqueue("independent:1", "GENERATION_CHANGED", independent,
                        GenerationOutbox.state(independent, 1, "SUCCEEDED"));
            });
            outbox.dispatch();
            assertThat(jdbc.queryForObject("SELECT attempts FROM afv_async_outbox WHERE event_key = 'failure:1'",
                    Integer.class)).isEqualTo(1);
            assertThat(jdbc.queryForObject(
                    "SELECT next_attempt_at > UTC_TIMESTAMP(6) FROM afv_async_outbox WHERE event_key = 'failure:1'",
                    Boolean.class)).isTrue();
            assertThat(jdbc.queryForObject("SELECT attempts FROM afv_async_outbox WHERE event_key = 'failure:2'",
                    Integer.class)).isZero();
            assertThat(jdbc.queryForObject("SELECT state FROM afv_async_outbox WHERE event_key = 'independent:1'",
                    String.class)).isEqualTo("DELIVERED");
        } finally {
            jdbc.execute("ALTER TABLE afv_generation_event DROP CHECK ck_test_outbox_failure");
        }
        jdbc.update("UPDATE afv_async_outbox SET next_attempt_at = UTC_TIMESTAMP(6) WHERE event_key = 'failure:1'");
        outbox.dispatch();
        assertThat(
                jdbc.queryForList("SELECT revision FROM afv_generation_event WHERE submission_id = ? ORDER BY sequence",
                        Long.class, broken)).containsExactly(1L, 2L);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_async_outbox WHERE state = 'PENDING'",
                Integer.class)).isZero();
    }

    @BeforeAll
    static void start() throws Exception {
        var source = new DriverManagerDataSource(MYSQL.getJdbcUrl(), MYSQL.getUsername(), MYSQL.getPassword());
        var flyway = Flyway.configure().dataSource(source).load();
        flyway.migrate();
        flyway.validate();
        jdbc = new JdbcTemplate(source);
        var manager = new DataSourceTransactionManager(source);
        transactions = new TransactionTemplate(manager);
        context = new AnnotationConfigApplicationContext();
        MybatisTestSupport.register(context, source, GenerationSubmissionMapper.class, RemoteGenerationJobMapper.class,
                GenerationAttemptMapper.class, AiModelMapper.class, ExecutionCoordinationMapper.class,
                AsyncOutboxMapper.class, MediaJobMapper.class, CanvasRunMapper.class, CanvasTextJobMapper.class);
        context.registerBean(JdbcTemplate.class, () -> jdbc);
        context.registerBean(PlatformTransactionManager.class, () -> manager);
        context.registerBean(TransactionTemplate.class, () -> transactions);
        context.registerBean(ObjectMapper.class, () -> new ObjectMapper().findAndRegisterModules());
        context.register(Transactions.class, CanvasJson.class, GenerationJobRepository.class,
                GenerationAttemptRepository.class, GenerationOutbox.class);
        context.refresh();
        jobs = context.getBean(GenerationJobRepository.class);
        attempts = context.getBean(GenerationAttemptRepository.class);
        outbox = context.getBean(GenerationOutbox.class);
        json = context.getBean(CanvasJson.class);
        jdbc.update("""
                INSERT INTO afv_ai_model(id, name, code, model_type, max_concurrency,
                    multimodal_input_types, multimodal_input_transports, reasoning_effort_levels)
                VALUES (900001, '测试模型', 'generation-job-test', 2, 1, JSON_ARRAY(), JSON_OBJECT(), JSON_ARRAY())
                """);
    }

    @BeforeEach
    void clear() {
        jdbc.update("DELETE FROM afv_generation_event");
        jdbc.update("DELETE FROM afv_async_outbox");
        jdbc.update("DELETE FROM afv_generation_attempt");
        jdbc.update("DELETE FROM afv_remote_generation_job");
        jdbc.update("DELETE FROM afv_generation_submission");
        jdbc.update("UPDATE afv_ai_model SET max_concurrency = 1 WHERE id = 900001");
    }

    @AfterAll
    static void stop() {
        if (context != null) {
            context.close();
        }
    }

    @Test
    void acceptedWorkIsInvisibleToWorkersUntilTransactionalOutboxDelivery() {
        String id = insert("QUEUED");
        assertThat(jobs.claim("worker-a")).isEmpty();
        transactions.executeWithoutResult(status -> outbox.enqueue("accepted:" + id, "GENERATION_ACCEPTED", id,
                GenerationOutbox.state(id, 1, "QUEUED")));
        assertThat(outbox.publishOne()).isTrue();
        assertThat(outbox.publishOne()).isFalse();
        assertThat(jobs.claim("worker-a")).isPresent();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_event WHERE submission_id = ?", Long.class,
                id)).isEqualTo(1);
    }

    @Test
    void failingAdmissionRollsBackAllRowsAndOutbox() {
        assertThatThrownBy(() -> transactions.executeWithoutResult(status -> {
            String id = insert("QUEUED");
            outbox.enqueue("accepted:" + id, "GENERATION_ACCEPTED", id, GenerationOutbox.state(id, 1, "QUEUED"));
            throw new IllegalStateException("受理末尾发生故障");
        })).isInstanceOf(IllegalStateException.class);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission", Long.class)).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_remote_generation_job", Long.class)).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_async_outbox", Long.class)).isZero();
    }

    @Test
    void competingInstancesCannotExceedModelPermitAndWaitingKeepsPermit() throws Exception {
        insert("READY");
        insert("READY");
        try (var executor = Executors.newFixedThreadPool(2)) {
            var start = new CountDownLatch(1);
            var first = executor.submit(() -> {
                start.await();
                return jobs.claim("worker-a");
            });
            var second = executor.submit(() -> {
                start.await();
                return jobs.claim("worker-b");
            });
            start.countDown();
            var claimed = List.of(first.get(10, TimeUnit.SECONDS), second.get(10, TimeUnit.SECONDS))
                    .stream().flatMap(value -> value.stream()).toList();
            assertThat(claimed).hasSize(1);
            var job = claimed.getFirst();
            jobs.waitForNextPoll(job.claim(), 60, "POLLING", null, null);
            jdbc.update("UPDATE afv_remote_generation_job SET next_poll_at = UTC_TIMESTAMP(6) WHERE state = 'READY'");
            assertThat(jobs.claim("worker-c")).isEmpty();
            assertThat(jobs.get(job.id()).permitHeld()).isTrue();
        }
    }

    @Test
    void restartFencesOldOwnerButAllowsImmutableLateReceiptWithoutResubmission() {
        String id = insert("READY");
        var old = jobs.claim("old-instance").orElseThrow();
        var intent = attempts.begin(old.claim(), GenerationPreparedRequest.ordinary());
        jdbc.update(
                "UPDATE afv_remote_generation_job SET lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?",
                old.id());
        var resumed = jobs.claim("new-instance").orElseThrow();
        assertThat(resumed.epoch()).isGreaterThan(old.epoch());
        assertThatThrownBy(() -> jobs.check(old.claim())).isInstanceOf(GenerationJobException.class);
        assertThatThrownBy(
                () -> attempts.begin(resumed.claim(), GenerationPreparedRequest.ordinary())).hasMessageContaining(
                "已提交");
        attempts.acknowledge(intent.id(), GenerationProviderReceipt.accepted("provider-accepted-once"));
        attempts.acknowledge(intent.id(), GenerationProviderReceipt.accepted("provider-accepted-once"));
        assertThatThrownBy(
                () -> attempts.acknowledge(intent.id(), GenerationProviderReceipt.accepted("different-task")))
                .isInstanceOf(GenerationJobException.class);
        var continued = attempts.applyReceipt(resumed.claim());
        assertThat(continued.phase()).isEqualTo("POLLING");
        assertThat(continued.providerTaskId()).isEqualTo("provider-accepted-once");
        assertThat(jobs.jobs(id)).hasSize(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_attempt", Long.class)).isEqualTo(1);
    }

    @Test
    void largeInlineImageResultsDoNotBlockOtherJobsOrReacquireCapacityDuringIngestRecovery() throws IOException {
        insert("READY");
        var processing = jobs.claim("ingest-a").orElseThrow();
        // 真实图片响应可达数 MB；队列排序不能携带完整 Base64 结果耗尽 MySQL 排序内存。
        String imageData = largeInlineImage();
        jobs.remoteResult(processing.claim(), RemoteGenerationResult.success(
                List.of(RemoteGenerationResult.Output.image(imageData))));
        String second = insert("READY");
        var next = jobs.claim("second-provider").orElseThrow();
        assertThat(next.submissionId()).isEqualTo(second);
        jdbc.update(
                "UPDATE afv_remote_generation_job SET lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?",
                processing.id());
        var resumed = jobs.claim("ingest-b").orElseThrow();
        assertThat(resumed.id()).isEqualTo(processing.id());
        assertThat(resumed.permitHeld()).isFalse();
        assertThat(resumed.result().outputs()).extracting(RemoteGenerationResult.Output::media)
                .containsExactly(imageData);
    }

    private String largeInlineImage() throws IOException {
        var image = new BufferedImage(1024, 1024, BufferedImage.TYPE_INT_RGB);
        var random = new Random(23);
        int[] pixels = new int[1024 * 1024];
        for (int index = 0; index < pixels.length; index++) {
            pixels[index] = random.nextInt(0x1000000);
        }
        image.setRGB(0, 0, 1024, 1024, pixels, 0, 1024);
        var bytes = new ByteArrayOutputStream();
        ImageIO.write(image, "png", bytes);
        return "data:image/png;base64," + Base64.getEncoder().encodeToString(bytes.toByteArray());
    }

    @Test
    void unknownSubmissionKeepsRemotePermitAndCannotBeAutomaticallyReclaimed() {
        insert("READY");
        var ambiguous = jobs.claim("worker-a").orElseThrow();
        attempts.begin(ambiguous.claim(), GenerationPreparedRequest.ordinary());
        jdbc.update(
                "UPDATE afv_remote_generation_job SET state = 'UNKNOWN_SUBMISSION', lease_until = NULL WHERE id = ?",
                ambiguous.id());
        insert("READY");
        assertThat(jobs.claim("worker-b")).isEmpty();
        assertThat(jobs.get(ambiguous.id()).permitHeld()).isTrue();
    }

    @Test
    void preparationCanResumeWithTheSameAttemptButCannotAcknowledgeBeforePaidIntent() {
        insert("READY");
        var first = jobs.claim("first-worker").orElseThrow();
        var prepared = attempts.prepare(first.claim());
        assertThat(prepared.submitted()).isFalse();
        assertThatThrownBy(() -> attempts.acknowledge(prepared.id(), GenerationProviderReceipt.accepted("not-sent")))
                .isInstanceOf(GenerationJobException.class);
        jdbc.update(
                "UPDATE afv_remote_generation_job SET lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?",
                first.id());
        var resumed = jobs.claim("replacement-worker").orElseThrow();
        assertThat(attempts.prepare(resumed.claim()).id()).isEqualTo(prepared.id());
        var submitted = attempts.begin(resumed.claim(), GenerationPreparedRequest.ordinary());
        assertThat(submitted.id()).isEqualTo(prepared.id());
        assertThat(submitted.submitted()).isTrue();
        attempts.acknowledge(submitted.id(), GenerationProviderReceipt.accepted("sent-once"));
        assertThat(attempts.applyReceipt(resumed.claim()).providerTaskId()).isEqualTo("sent-once");
    }

    @Test
    void lateReceiptDuringPreparationBlocksASecondPaidIntent() {
        insert("READY");
        var job = jobs.claim("worker").orElseThrow();
        var first = attempts.begin(job.claim(), GenerationPreparedRequest.ordinary());
        jdbc.update("UPDATE afv_generation_attempt SET resolution = 'CONFIRM_NOT_ACCEPTED_AND_RETRY' WHERE id = ?",
                first.id());
        jdbc.update(
                "UPDATE afv_remote_generation_job SET phase = 'NEW', active_attempt_id = NULL, provider_task_id = NULL WHERE id = ?",
                job.id());
        var next = attempts.prepare(job.claim());
        attempts.acknowledge(first.id(), GenerationProviderReceipt.accepted("original-provider-task"));
        assertThatThrownBy(() -> attempts.begin(job.claim(), GenerationPreparedRequest.ordinary()))
                .isInstanceOf(GenerationJobException.class).hasMessageContaining("迟到回执");
        assertThat(attempts.get(next.id()).submitted()).isFalse();
        assertThat(jobs.get(job.id()).phase()).isEqualTo("PREPARING");
    }

    private String insert(String state) {
        String id = UUID.randomUUID().toString();
        var actor = ResourceCommandContext.user(new CanvasActor(101, null), null);
        var model = AiModel.builder().id(900001L).modelType(2).code("generation-job-test").build();
        jdbc.update("""
                        INSERT INTO afv_generation_submission(id, request_id, scope_key, request_hash, kind, task_id, public_task_id,
                            actor_user_id, context_json, model_snapshot_json, api_config_id, api_config_hash, state, created_at, updated_at)
                        VALUES (?, ?, ?, ?, 'IMAGE', ?, ?, 101, ?, ?, 1, ?, 'QUEUED', UTC_TIMESTAMP(6), UTC_TIMESTAMP(6))
                        """, id, id, "s".repeat(64), "h".repeat(64), Math.abs(UUID.randomUUID().getMostSignificantBits()), id,
                json.write(actor), json.write(model), "a".repeat(64));
        jdbc.update("""
                        INSERT INTO afv_remote_generation_job(id, submission_id, item_id, item_index, model_id, actor_user_id,
                            state, phase, input_json, next_poll_at, deadline_at, created_at, updated_at)
                        VALUES (?, ?, 1, 0, 900001, 101, ?, 'NEW', ?, UTC_TIMESTAMP(6), TIMESTAMPADD(HOUR, 1, UTC_TIMESTAMP(6)),
                            UTC_TIMESTAMP(6), UTC_TIMESTAMP(6))
                        """, UUID.randomUUID().toString(), id, state,
                json.write(Map.of("prompt", "测试", "_resultItemIds", List.of(1L))));
        return id;
    }
}
