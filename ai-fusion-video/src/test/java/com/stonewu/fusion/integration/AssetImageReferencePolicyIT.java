package com.stonewu.fusion.integration;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.RemoteGenerationJobEntity;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.generation.RemoteGenerationJobMapper;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.generation.job.AssetImageReferencePolicy;
import com.stonewu.fusion.service.generation.job.GenerationJobException;
import com.stonewu.fusion.service.generation.job.GenerationRequestRepository;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.annotation.EnableTransactionManagement;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 使用 MySQL 验证首次输入的持久化、运行隔离和并发受理，不调用图片供应商。 */
@Testcontainers
class AssetImageReferencePolicyIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("asset_reference_test").withUsername("afv").withPassword("afv-test");
    static final AtomicLong IDS = new AtomicLong(60000);
    static final String MEDIA_ID = "00000000-0000-0000-0000-000000000001";
    static AnnotationConfigApplicationContext application;
    static TransactionTemplate transactions;
    static CanvasJson json;
    static AgentRunMapper runs;
    static GenerationSubmissionMapper submissions;
    static RemoteGenerationJobMapper jobs;
    static GenerationRequestRepository requests;
    static ResourceAgentFence fence;
    static AssetImageReferencePolicy policy;

    @Configuration
    @EnableTransactionManagement
    static class Transactions {
    }

    @BeforeAll
    static void start() throws Exception {
        var source = new DriverManagerDataSource(MYSQL.getJdbcUrl(), MYSQL.getUsername(), MYSQL.getPassword());
        var flyway = Flyway.configure().dataSource(source).load();
        flyway.migrate();
        flyway.validate();
        var manager = new DataSourceTransactionManager(source);
        transactions = new TransactionTemplate(manager);
        transactions.setIsolationLevel(TransactionDefinition.ISOLATION_READ_COMMITTED);
        application = new AnnotationConfigApplicationContext();
        MybatisTestSupport.register(application, source, AgentRunMapper.class, GenerationSubmissionMapper.class,
                RemoteGenerationJobMapper.class);
        application.registerBean(PlatformTransactionManager.class, () -> manager);
        application.registerBean(ObjectMapper.class, () -> new ObjectMapper().findAndRegisterModules());
        application.registerBean(GenerationRequestRepository.class,
                () -> new GenerationRequestRepository(null, null, application.getBean(CanvasJson.class)));
        application.registerBean(ReferenceImageTransportService.class,
                () -> new ReferenceImageTransportService(null, null, null, null));
        application.register(Transactions.class, CanvasJson.class, ResourceAgentFence.class,
                AssetImageReferencePolicy.class);
        application.refresh();
        json = application.getBean(CanvasJson.class);
        runs = application.getBean(AgentRunMapper.class);
        submissions = application.getBean(GenerationSubmissionMapper.class);
        jobs = application.getBean(RemoteGenerationJobMapper.class);
        requests = application.getBean(GenerationRequestRepository.class);
        fence = application.getBean(ResourceAgentFence.class);
        policy = application.getBean(AssetImageReferencePolicy.class);
    }

    @AfterAll
    static void stop() {
        if (application != null) {
            application.close();
        }
    }

    @Test
    void failedGenerationKeepsItsReferencesAfterWorkerReplacement() {
        var original = run(IDS.incrementAndGet());
        var first = transactions.execute(status -> admit(input(true), original));
        submissions.update(new LambdaUpdateWrapper<GenerationSubmissionEntity>()
                .eq(GenerationSubmissionEntity::getId, first).set(GenerationSubmissionEntity::getState, "FAILED"));
        runs.update(new LambdaUpdateWrapper<AgentRun>().eq(AgentRun::getRunId, original.agentRunId())
                .set(AgentRun::getOwnerInstanceId, "worker-b").set(AgentRun::getOwnerEpoch, 2));
        var restored = new ResourceCommandContext(original.actor(), "agent", null, original.agentRunId(),
                UUID.randomUUID().toString(), "worker-b", 2L);

        assertThatThrownBy(() -> transactions.execute(status -> admit(input(false), restored)))
                .isInstanceOfSatisfying(GenerationJobException.class,
                        failure -> assertThat(failure.getErrorCode()).isEqualTo("ASSET_IMAGE_REFERENCES_CHANGED"));
        assertThat(count(original)).isEqualTo(1);

        var retry = input(true);
        retry.setInputMediaVersions(null);
        transactions.executeWithoutResult(status -> {
            fence.requireActive(restored, null);
            policy.enforce(retry, restored);
        });
        assertThat(retry.getInputMediaVersions()).containsExactlyEntriesOf(Map.of(MEDIA_ID, 3));
    }

    @Test
    void differentAssetExecutorsDoNotInheritEachOthersReferences() {
        long userId = IDS.incrementAndGet();
        var first = run(userId);
        var sameUserOtherRun = run(userId);
        var otherUser = run(IDS.incrementAndGet());
        transactions.executeWithoutResult(status -> admit(input(true), first));

        transactions.executeWithoutResult(status -> admit(input(false), sameUserOtherRun));
        transactions.executeWithoutResult(status -> admit(input(false), otherUser));

        assertThat(count(first)).isEqualTo(2);
        assertThat(count(otherUser)).isEqualTo(1);
        assertThat(submissions.selectAssetImageReferenceSnapshot(requests.scope(sameUserOtherRun),
                sameUserOtherRun.agentRunId()).inputJson()).contains("没有参考图的初始形象");
    }

    @Test
    void concurrentInvocationWaitsForTheFirstReferenceSnapshot() throws Exception {
        var original = run(IDS.incrementAndGet());
        var retry = new ResourceCommandContext(original.actor(), "agent", null, original.agentRunId(),
                UUID.randomUUID().toString(), "worker-a", 1L);
        var locked = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var secondStarted = new CountDownLatch(1);
        try (var pool = Executors.newFixedThreadPool(2)) {
            var first = pool.submit(() -> transactions.execute(status -> {
                String accepted = admit(input(true), original);
                locked.countDown();
                await(release);
                return accepted;
            }));
            assertThat(locked.await(10, TimeUnit.SECONDS)).isTrue();
            var second = pool.submit(() -> {
                secondStarted.countDown();
                try {
                    transactions.execute(status -> admit(input(false), retry));
                    return "accepted";
                } catch (GenerationJobException failure) {
                    return failure.getErrorCode();
                }
            });
            try {
                assertThat(secondStarted.await(10, TimeUnit.SECONDS)).isTrue();
                assertThatThrownBy(() -> second.get(200, TimeUnit.MILLISECONDS)).isInstanceOf(TimeoutException.class);
            } finally {
                release.countDown();
            }
            assertThat(first.get(10, TimeUnit.SECONDS)).isNotBlank();
            assertThat(second.get(10, TimeUnit.SECONDS)).isEqualTo("ASSET_IMAGE_REFERENCES_CHANGED");
        }
        assertThat(count(original)).isEqualTo(1);
    }

    private static ResourceCommandContext run(long userId) {
        String id = UUID.randomUUID().toString();
        var now = LocalDateTime.now(ZoneOffset.UTC);
        runs.insert(AgentRun.builder().runId(id).conversationId(id).userId(userId)
                .agentType("asset_image_executor").kernelFingerprint("test").agentDefinitionSnapshotJson("{}")
                .agentStateSessionId(id).status("RUNNING").ownerInstanceId("worker-a").ownerEpoch(1L)
                .leaseUntil(now.plusMinutes(10)).deadlineAt(now.plusMinutes(10)).startedAt(now).build());
        return new ResourceCommandContext(new CanvasActor(userId, null), "agent", null, id,
                UUID.randomUUID().toString(), "worker-a", 1L);
    }

    private static ImageTask input(boolean withReferences) {
        return ImageTask.builder().prompt(withReferences ? "有角色参考图的变体" : "没有参考图的初始形象")
                .refImageUrls(withReferences ? json.write(List.of("http://localhost:3000/api/art-styles/realistic.jpg",
                        "media://" + MEDIA_ID)) : null)
                .inputMediaVersions(withReferences ? Map.of(MEDIA_ID, 3) : Map.of()).build();
    }

    private static String admit(ImageTask task, ResourceCommandContext context) {
        fence.requireActive(context, null);
        policy.enforce(task, context);
        String id = UUID.randomUUID().toString();
        long taskId = IDS.incrementAndGet();
        submissions.insertQueued(GenerationSubmissionEntity.builder().id(id).requestId(context.toolCallId())
                .scopeKey(requests.scope(context)).requestHash(json.hash(task)).kind("IMAGE").taskId(taskId)
                .publicTaskId(UUID.randomUUID().toString()).actorUserId(context.actor().userId())
                .contextJson(json.write(context)).modelSnapshotJson("{}").apiConfigId(1L)
                .apiConfigHash(json.hash(Map.of())).build());
        jobs.insertQueued(RemoteGenerationJobEntity.builder().id(UUID.randomUUID().toString())
                .submissionId(id).itemId(taskId).itemIndex(0).modelId(1L).actorUserId(context.actor().userId())
                .inputJson(json.write(task)).build());
        return id;
    }

    private static long count(ResourceCommandContext context) {
        return submissions.selectCount(new LambdaQueryWrapper<GenerationSubmissionEntity>()
                .eq(GenerationSubmissionEntity::getScopeKey, requests.scope(context)));
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(10, TimeUnit.SECONDS)) {
                throw new IllegalStateException("等待并发测试释放事务超时");
            }
        } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(failure);
        }
    }
}
