package com.stonewu.fusion.service.ai.text;

import com.stonewu.fusion.entity.ai.AgentWorkspaceEntry;
import com.stonewu.fusion.mapper.ai.AgentWorkspaceEntryMapper;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceBackend;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceBaseStore;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceConfigService;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceLocation;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspacePayloadService;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceStoredPayload;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoSpyBean;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionSynchronization;
import org.springframework.transaction.support.TransactionSynchronizationManager;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mockingDetails;

@SpringBootTest
@Testcontainers(disabledWithoutDocker = false)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class AgentLongContentPersistenceTests {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container
    static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "long-content-" + UUID.randomUUID()).toAbsolutePath();

    @DynamicPropertySource
    static void isolatedProperties(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url", MYSQL::getJdbcUrl);
        properties.add("spring.datasource.username", MYSQL::getUsername);
        properties.add("spring.datasource.password", MYSQL::getPassword);
        properties.add("spring.data.redis.host", REDIS::getHost);
        properties.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        properties.add("spring.data.redis.password", () -> "");
        properties.add("app.storage.local-base-path", FILES::toString);
        properties.add("app.agent-workspace.local-base-path", FILES::toString);
    }

    @Autowired
    LongToolResultStore results;
    @Autowired
    AgentWorkspaceBaseStore workspace;
    @Autowired
    AgentWorkspaceConfigService configuration;
    @Autowired
    PlatformTransactionManager transactions;
    @MockitoSpyBean
    AgentWorkspaceEntryMapper entries;
    @MockitoSpyBean
    AgentWorkspacePayloadService payloads;

    @AfterEach
    void resetLocation() {
        configuration.activate(new AgentWorkspaceLocation(AgentWorkspaceBackend.DATABASE, null, null), null);
    }

    @Test
    @DisplayName("慢文件写入不占用数据库锁，并发保存同一结果只登记一份有效正文")
    void slowConcurrentUploadsReleaseTheDatabaseAndKeepOneImmutableArtifact() throws Exception {
        configuration.activate(new AgentWorkspaceLocation(AgentWorkspaceBackend.LOCAL, null, FILES.toString()), null);
        CountDownLatch writing = new CountDownLatch(2);
        CountDownLatch release = new CountDownLatch(1);
        List<AgentWorkspaceStoredPayload> staged = new CopyOnWriteArrayList<>();
        doAnswer(call -> {
            assertThat(TransactionSynchronizationManager.isActualTransactionActive()).isFalse();
            writing.countDown();
            assertThat(release.await(15, TimeUnit.SECONDS)).isTrue();
            AgentWorkspaceStoredPayload stored = (AgentWorkspaceStoredPayload) call.callRealMethod();
            staged.add(stored);
            return stored;
        }).when(payloads).write(anyString(), any());
        String session = UUID.randomUUID().toString();
        String content = "首".repeat(2000) + "中".repeat(91910) + "尾".repeat(2000);
        var first = CompletableFuture.supplyAsync(() -> results.save("7", session, content));
        var second = CompletableFuture.supplyAsync(() -> results.save("7", session, content));
        try {
            assertThat(writing.await(10, TimeUnit.SECONDS)).isTrue();
            // 另一个连接必须能拿到同一配置行的写锁，避免只测试不受行锁影响的普通 SELECT。
            assertThat(CompletableFuture.supplyAsync(configuration::lockWritableLocation).get(3, TimeUnit.SECONDS)
                    .backendType()).isEqualTo(AgentWorkspaceBackend.LOCAL);
        } finally {
            release.countDown();
        }
        String id = first.get(15, TimeUnit.SECONDS);
        assertThat(second.get(15, TimeUnit.SECONDS)).isEqualTo(id);
        var reopened = new LongToolResultStore(workspace);
        StringBuilder restored = new StringBuilder();
        int offset = 0;
        do {
            var page = reopened.read("7", session, id, new TextPages.Request(offset, 8000, id));
            restored.append(page.content());
            offset = page.nextOffset() == null ? -1 : page.nextOffset();
        } while (offset >= 0);
        assertThat(restored.toString()).isEqualTo(content);
        assertThat(staged).hasSize(2);
        assertThat(staged.stream().filter(item -> Files.exists(FILES.resolve(item.contentRef())))).hasSize(1);
    }

    @Test
    @DisplayName("上传期间切换存储时拒绝迟到提交，并清理未登记文件")
    void changedStorageLocationRejectsLateUpload() {
        configuration.activate(new AgentWorkspaceLocation(AgentWorkspaceBackend.LOCAL, null, FILES.toString()), null);
        List<AgentWorkspaceStoredPayload> staged = new CopyOnWriteArrayList<>();
        doAnswer(call -> {
            configuration.activate(new AgentWorkspaceLocation(AgentWorkspaceBackend.DATABASE, null, null), null);
            AgentWorkspaceStoredPayload stored = (AgentWorkspaceStoredPayload) call.callRealMethod();
            staged.add(stored);
            return stored;
        }).when(payloads).write(anyString(), any());
        String session = UUID.randomUUID().toString();
        assertThatThrownBy(() -> results.save("7", session, "未提交的正文"))
                .hasMessageContaining("存储已变化");
        assertThat(workspace.find(List.of("tool-results", "7", TextPages.version(session)),
                TextPages.version("未提交的正文"))).isNull();
        assertThat(staged).singleElement().satisfies(item ->
                assertThat(Files.exists(FILES.resolve(item.contentRef()))).isFalse());
    }

    @Test
    @DisplayName("拒绝带外层事务的文件写入，数据库存储的结果可由新读取器恢复")
    void rejectsAnOuterTransactionAndReopensDatabaseContent() {
        String session = UUID.randomUUID().toString();
        assertThatThrownBy(() -> new TransactionTemplate(transactions).execute(status ->
                results.save("7", session, "正文"))).hasMessageContaining("不能在数据库事务内");
        String id = results.save("7", session, "正文🌕");
        assertThat(new LongToolResultStore(workspace).read("7", session, id,
                new TextPages.Request(0, 4000, null)).content()).isEqualTo("正文🌕");
    }

    @Test
    @DisplayName("数据库已提交但回执丢失时不能删除已登记正文，再次读取仍然完整")
    void lostCommitAcknowledgementDoesNotDeleteCommittedContent() {
        configuration.activate(new AgentWorkspaceLocation(AgentWorkspaceBackend.LOCAL, null, FILES.toString()), null);
        AtomicBoolean acknowledgementLost = new AtomicBoolean();
        var insertDelegate = mockingDetails(entries).getMockCreationSettings().getDefaultAnswer();
        doAnswer(call -> {
            Object inserted = insertDelegate.answer(call);
            // 在正文登记事务提交完成后注入故障，确保测试覆盖已提交文件的保留行为。
            TransactionSynchronizationManager.registerSynchronization(new TransactionSynchronization() {
                @Override
                public void afterCommit() {
                    acknowledgementLost.set(true);
                    throw new IllegalStateException("提交回执丢失");
                }
            });
            return inserted;
        }).when(entries).insert(any(AgentWorkspaceEntry.class));
        String session = UUID.randomUUID().toString();
        String content = "已经完整提交的正文";
        assertThatThrownBy(() -> results.save("7", session, content)).hasMessageContaining("提交回执丢失");
        assertThat(acknowledgementLost).isTrue();
        String id = results.save("7", session, content);
        assertThat(results.read("7", session, id, new TextPages.Request(0, 4000, null)).content()).isEqualTo(content);
    }
}
