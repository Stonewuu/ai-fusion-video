package com.stonewu.fusion.service.ai.agentscope;

import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelKey;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelModelFactory;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelResource;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolkitResources;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolRegistry;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentSessionSerialGate;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentScopeHarnessFactory;
import com.stonewu.fusion.service.ai.agentscope.kernel.DefaultAgentScopeHarnessInvoker;
import com.stonewu.fusion.service.ai.agentscope.kernel.HarnessLeaseCache;
import com.stonewu.fusion.service.ai.agentscope.kernel.OwnedChatModel;
import com.stonewu.fusion.service.ai.agentscope.message.AgentScopeMessageMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.AgentScopeStateStoreFactory;
import com.stonewu.fusion.service.ai.agentscope.state.AgentScopeShutdownRecoveryBridge;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStatePreflight;
import com.stonewu.fusion.service.ai.agentscope.state.FailClosedAgentStateStore;
import com.stonewu.fusion.service.ai.agentscope.state.InMemoryStateStoreFailureGuard;
import com.stonewu.fusion.service.ai.agentscope.context.ToolPermissionContext;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEventType;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.state.AgentState;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.InMemoryAgentStateStore;
import io.agentscope.core.state.State;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.ConfigDataApplicationContextInitializer;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;

class AgentScopeNoToolKernelIntegrationTests {

    @Test
    void sameSlotFreshCallWaitsForRecoveryAcknowledgement() throws InterruptedException {
        AgentScopeRuntimeProperties properties = singleThreadProperties();
        properties.setStateThreads(2);
        InMemoryStateStoreFailureGuard failures = new InMemoryStateStoreFailureGuard();
        AckGatedCopyingStateStore backingStore = new AckGatedCopyingStateStore();
        AgentStateStore store = new FailClosedAgentStateStore(backingStore, failures);
        EchoModel model = new EchoModel();

        try (AgentRuntimeSchedulers schedulers = new AgentRuntimeSchedulers(properties)) {
            AgentScopeHarnessFactory harnessFactory = new AgentScopeHarnessFactory(
                    ignored -> OwnedChatModel.owned(model),
                    (ignored, toolkit) -> AgentKernelToolkitResources.none(),
                    store,
                    failures,
                    new AgentScopeShutdownRecoveryBridge(store, schedulers));
            AgentScopeV2Properties cacheProperties = new AgentScopeV2Properties();
            cacheProperties.getCache().setMaximumSize(4);
            cacheProperties.getCache().setExpireAfterAccess(Duration.ofMinutes(30));
            cacheProperties.getCache().setCapacityWait(Duration.ofSeconds(1));
            HarnessLeaseCache cache = new HarnessLeaseCache(
                    harnessFactory, schedulers, cacheProperties);
            DefaultAgentScopeHarnessInvoker invoker = new DefaultAgentScopeHarnessInvoker(
                    cache,
                    new AgentStatePreflight(store, failures, schedulers),
                    new AgentSessionSerialGate(),
                    schedulers);
            AgentScopeMessageMapper messages = new AgentScopeMessageMapper();
            AgentKernelSpec spec = noToolSpec();
            RuntimeContext initialContext = context("42", "afv:v2:recovery-race:no-tool");

            StepVerifier.create(invoker.call(
                            spec, messages.toUserMessages("first echo"), initialContext))
                    .assertNext(response ->
                            assertThat(response.getTextContent()).isEqualTo("first echo"))
                    .verifyComplete();

            AgentState interrupted = store.get(
                            initialContext.getUserId(),
                            initialContext.getSessionId(),
                            "agent_state",
                            AgentState.class)
                    .orElseThrow();
            interrupted.setShutdownInterrupted(true);
            store.save(
                    initialContext.getUserId(),
                    initialContext.getSessionId(),
                    "agent_state",
                    interrupted);
            backingStore.armAcknowledgementGate();

            Mono<Msg> recovery = invoker.call(
                            spec,
                            messages.toUserMessages("retried first echo"),
                            context("42", "afv:v2:recovery-race:no-tool"))
                    .cache();
            Mono<Msg> fresh = invoker.call(
                            spec,
                            messages.toUserMessages("fresh after acknowledgement"),
                            context("42", "afv:v2:recovery-race:no-tool"))
                    .cache();
            AtomicReference<Throwable> asynchronousFailure = new AtomicReference<>();

            recovery.subscribe(ignored -> { }, asynchronousFailure::set);
            assertThat(backingStore.awaitAcknowledgementSave(Duration.ofSeconds(2))).isTrue();
            try {
                fresh.subscribe(ignored -> { }, asynchronousFailure::set);
                assertThat(model.thirdCallStarted.await(250, TimeUnit.MILLISECONDS)).isFalse();
            } finally {
                backingStore.releaseAcknowledgementSave();
            }

            StepVerifier.create(Mono.zip(recovery, fresh))
                    .assertNext(results -> {
                        assertThat(results.getT1().getTextContent()).isEqualTo("first echo");
                        assertThat(results.getT2().getTextContent())
                                .isEqualTo("fresh after acknowledgement");
                    })
                    .verifyComplete();
            assertThat(asynchronousFailure).hasNullValue();
            assertThat(model.messagesByCall).hasSize(3);
            assertThat(model.messagesByCall.get(1))
                    .extracting(Msg::getTextContent)
                    .doesNotContain("retried first echo");
            assertThat(model.messagesByCall.get(2))
                    .extracting(Msg::getTextContent)
                    .contains("fresh after acknowledgement");
            assertThat(store.get(
                            initialContext.getUserId(),
                            initialContext.getSessionId(),
                            "agent_state",
                            AgentState.class)
                    .orElseThrow()
                    .isShutdownInterrupted()).isFalse();

            StepVerifier.create(cache.drainAndClose(Duration.ofSeconds(2))).verifyComplete();
            assertThat(model.closeCount).hasValue(1);
        } finally {
            backingStore.releaseAcknowledgementSave();
            store.close();
        }
    }

    @Test
    void callsAndStreamsThroughOneNoToolKernelWithIsolatedStateSlots() {
        AgentScopeRuntimeProperties properties = singleThreadProperties();
        InMemoryStateStoreFailureGuard failures = new InMemoryStateStoreFailureGuard();
        EchoModel model = new EchoModel();
        AtomicInteger registryCalls = new AtomicInteger();

        AgentStateStore store = new AgentScopeStateStoreFactory(failures).createInMemory();
        try (AgentRuntimeSchedulers schedulers = new AgentRuntimeSchedulers(properties)) {
            AgentScopeHarnessFactory harnessFactory = new AgentScopeHarnessFactory(
                    ignored -> OwnedChatModel.owned(model),
                    (ignored, toolkit) -> {
                        registryCalls.incrementAndGet();
                        return AgentKernelToolkitResources.none();
                    },
                    store,
                    failures,
                    new AgentScopeShutdownRecoveryBridge(store, schedulers));
            AgentScopeV2Properties cacheProperties = new AgentScopeV2Properties();
            cacheProperties.getCache().setMaximumSize(4);
            cacheProperties.getCache().setExpireAfterAccess(Duration.ofMinutes(30));
            cacheProperties.getCache().setCapacityWait(Duration.ofSeconds(1));
            HarnessLeaseCache cache = new HarnessLeaseCache(
                    harnessFactory, schedulers, cacheProperties);
            DefaultAgentScopeHarnessInvoker invoker = new DefaultAgentScopeHarnessInvoker(
                    cache,
                    new AgentStatePreflight(store, failures, schedulers),
                    new AgentSessionSerialGate(),
                    schedulers);
            AgentScopeMessageMapper messages = new AgentScopeMessageMapper();
            AgentKernelSpec spec = noToolSpec();
            RuntimeContext firstContext = context("42", "afv:v2:conversation-a:no-tool");
            RuntimeContext secondContext = context("42", "afv:v2:conversation-b:no-tool");

            assertThat(store.exists(firstContext.getUserId(), firstContext.getSessionId())).isFalse();
            assertThat(store.exists(secondContext.getUserId(), secondContext.getSessionId())).isFalse();

            StepVerifier.create(invoker.call(
                            spec, messages.toUserMessages("first echo"), firstContext))
                    .assertNext(response ->
                            assertThat(response.getTextContent()).isEqualTo("first echo"))
                    .verifyComplete();

            assertThat(store.exists(firstContext.getUserId(), firstContext.getSessionId())).isTrue();
            assertThat(store.exists(secondContext.getUserId(), secondContext.getSessionId())).isFalse();

            AgentState interrupted = store.get(
                            firstContext.getUserId(),
                            firstContext.getSessionId(),
                            "agent_state",
                            AgentState.class)
                    .orElseThrow();
            interrupted.setShutdownInterrupted(true);
            store.save(
                    firstContext.getUserId(),
                    firstContext.getSessionId(),
                    "agent_state",
                    interrupted);
            StepVerifier.create(invoker.call(
                            spec, messages.toUserMessages("retried first echo"), firstContext))
                    .assertNext(response ->
                            assertThat(response.getTextContent()).isEqualTo("first echo"))
                    .verifyComplete();
            assertThat(store.get(
                            firstContext.getUserId(),
                            firstContext.getSessionId(),
                            "agent_state",
                            AgentState.class)
                    .orElseThrow()
                    .isShutdownInterrupted()).isFalse();
            assertThat(model.messagesByCall.get(1))
                    .extracting(Msg::getTextContent)
                    .doesNotContain("retried first echo");

            StepVerifier.create(invoker.streamEvents(
                            spec, messages.toUserMessages("second echo"), secondContext)
                    .map(event -> event.getType())
                    .collectList())
                    .assertNext(types -> assertThat(types)
                            .containsSubsequence(
                                    AgentEventType.AGENT_START,
                                    AgentEventType.MODEL_CALL_START,
                                    AgentEventType.TEXT_BLOCK_DELTA,
                                    AgentEventType.MODEL_CALL_END,
                                    AgentEventType.AGENT_END))
                    .verifyComplete();

            assertThat(registryCalls).hasValue(1);
            assertThat(model.toolsByCall).hasSize(3)
                    .allSatisfy(tools -> assertThat(tools).isEmpty());
            assertThat(store.exists(secondContext.getUserId(), secondContext.getSessionId())).isTrue();
            assertStateSlot(store, firstContext);
            assertStateSlot(store, secondContext);
            assertThat(model.messagesByCall).hasSize(3);
            assertThat(model.messagesByCall.get(2))
                    .extracting(Msg::getTextContent)
                    .doesNotContain("first echo")
                    .contains("second echo");
            store.delete(firstContext.getUserId(), firstContext.getSessionId());
            assertThat(store.exists(firstContext.getUserId(), firstContext.getSessionId())).isFalse();
            assertThat(store.exists(secondContext.getUserId(), secondContext.getSessionId())).isTrue();

            StepVerifier.create(cache.drainAndClose(Duration.ofSeconds(2))).verifyComplete();
            assertThat(model.closeCount).hasValue(1);
        } finally {
            store.close();
        }
    }

    @Test
    void localYamlBindsAndWiresTheRealSpringRuntime() {
        EchoModel model = new EchoModel();
        AtomicInteger registryCalls = new AtomicInteger();
        new ApplicationContextRunner()
                .withInitializer(new ConfigDataApplicationContextInitializer())
                .withPropertyValues(
                        "fusion.agentscope.v2.state.mode=IN_MEMORY",
                        "app.agentscope.runtime.state-threads=1",
                        "app.agentscope.runtime.journal-threads=1",
                        "app.agentscope.runtime.model-threads=1",
                        "app.agentscope.runtime.tool-threads=1")
                .withBean(AgentKernelModelFactory.class,
                        () -> ignored -> OwnedChatModel.owned(model))
                .withBean(AgentKernelToolRegistry.class, () -> (ignored, toolkit) -> {
                    registryCalls.incrementAndGet();
                    return AgentKernelToolkitResources.none();
                })
                .withUserConfiguration(
                        com.stonewu.fusion.config.AgentScopeRuntimeConfiguration.class,
                        AgentSessionSerialGate.class,
                        AgentScopeHarnessFactory.class,
                        AgentScopeShutdownRecoveryBridge.class,
                        HarnessLeaseCache.class,
                        DefaultAgentScopeHarnessInvoker.class)
                .run(context -> {
                    assertThat(context.getStartupFailure()).isNull();
                    AgentScopeV2Properties properties = context.getBean(AgentScopeV2Properties.class);
                    assertThat(properties.getCache().getMaximumSize()).isEqualTo(64);
                    assertThat(properties.getCache().getExpireAfterAccess()).isEqualTo(Duration.ofMinutes(30));
                    assertThat(properties.getCache().getCapacityWait()).isEqualTo(Duration.ofSeconds(5));
                    assertThat(properties.getState().getMode())
                            .isEqualTo(AgentScopeV2Properties.Mode.IN_MEMORY);
                    assertThat(properties.getState().getDatabaseName()).isEqualTo("ai_fusion_video");
                    assertThat(properties.getState().getTableName()).isEqualTo("afv_agent_state");
                    assertThat(properties.getSkills().isEnabled()).isTrue();
                    assertThat(properties.getSkills().getRepositories())
                            .containsKey("bundled");
                    assertThat(properties.getMcp().isEnabled()).isFalse();

                    RuntimeContext runtime = context("7", "afv:v2:spring-wiring:no-tool");
                    StepVerifier.create(context.getBean(DefaultAgentScopeHarnessInvoker.class).call(
                                    noToolSpec(),
                                    new AgentScopeMessageMapper().toUserMessages("spring echo"),
                                    runtime))
                            .assertNext(response ->
                                    assertThat(response.getTextContent()).isEqualTo("spring echo"))
                            .verifyComplete();

                    AgentStateStore store = context.getBean(AgentStateStore.class);
                    assertStateSlot(store, runtime);
                    assertThat(registryCalls).hasValue(1);
                    StepVerifier.create(context.getBean(HarnessLeaseCache.class)
                                    .drainAndClose(Duration.ofSeconds(2)))
                            .verifyComplete();
                    assertThat(model.closeCount).hasValue(1);
                });
    }

    private static void assertStateSlot(AgentStateStore store, RuntimeContext context) {
        AgentState state = store.get(
                        context.getUserId(), context.getSessionId(), "agent_state", AgentState.class)
                .orElseThrow();
        assertThat(state.getUserId()).isEqualTo(context.getUserId());
        assertThat(state.getSessionId()).isEqualTo(context.getSessionId());
    }

    private static AgentScopeRuntimeProperties singleThreadProperties() {
        AgentScopeRuntimeProperties properties = new AgentScopeRuntimeProperties();
        properties.setStateThreads(1);
        properties.setJournalThreads(1);
        properties.setModelThreads(1);
        properties.setToolThreads(1);
        return properties;
    }

    private static RuntimeContext context(String userId, String sessionId) {
        return RuntimeContext.builder()
                .userId(userId)
                .sessionId(sessionId)
                .put(
                        ToolPermissionContext.class,
                        new ToolPermissionContext(ToolExecutionMode.DEFAULT))
                .build();
    }

    private static AgentKernelSpec noToolSpec() {
        AiModel model = AiModel.builder().id(101L).code("echo-model").build();
        String stableKey = "no-tool-integration";
        String prompt = "Echo the user's text exactly.";
        String whitelistVersion = AgentKernelKey.whitelistVersion("afv-tools-v1", Set.of());
        AgentKernelKey key = AgentKernelKey.create(
                stableKey,
                AgentKernelKey.modelRecordFingerprint(model),
                AgentKernelKey.promptVersion(prompt),
                List.of(),
                whitelistVersion);
        return new AgentKernelSpec(
                key,
                model,
                stableKey,
                "No Tool Echo",
                "Echoes text without tools",
                prompt,
                3,
                List.of(),
                Set.of(),
                whitelistVersion);
    }

    private static final class EchoModel extends ChatModelBase implements AutoCloseable {
        private final AtomicInteger calls = new AtomicInteger();
        private final AtomicInteger closeCount = new AtomicInteger();
        private final CountDownLatch thirdCallStarted = new CountDownLatch(1);
        private final List<List<ToolSchema>> toolsByCall = new CopyOnWriteArrayList<>();
        private final List<List<Msg>> messagesByCall = new CopyOnWriteArrayList<>();

        @Override
        protected Flux<ChatResponse> doStream(
                List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            toolsByCall.add(List.copyOf(tools));
            messagesByCall.add(List.copyOf(messages));
            int callNumber = calls.incrementAndGet();
            if (callNumber == 3) {
                thirdCallStarted.countDown();
            }
            String replyId = "echo-" + callNumber;
            String text = messages.getLast().getTextContent();
            return Flux.just(
                    ChatResponse.builder()
                            .id(replyId)
                            .content(List.of(TextBlock.builder().text(text).build()))
                            .build(),
                    ChatResponse.builder()
                            .id(replyId)
                            .content(List.of())
                            .finishReason("stop")
                            .build());
        }

        @Override
        public String getModelName() {
            return "echo-model";
        }

        @Override
        public void close() {
            closeCount.incrementAndGet();
        }
    }

    private static final class AckGatedCopyingStateStore extends InMemoryAgentStateStore {
        private final AtomicBoolean gateArmed = new AtomicBoolean();
        private final CountDownLatch acknowledgementSaveEntered = new CountDownLatch(1);
        private final CountDownLatch releaseAcknowledgementSave = new CountDownLatch(1);

        @Override
        public void save(String userId, String sessionId, String key, State state) {
            State stateCopy = copy(state);
            if (state instanceof AgentState agentState
                    && "agent_state".equals(key)
                    && !agentState.isShutdownInterrupted()
                    && gateArmed.compareAndSet(true, false)) {
                acknowledgementSaveEntered.countDown();
                awaitRelease();
            }
            super.save(userId, sessionId, key, stateCopy);
        }

        @Override
        public <T extends State> Optional<T> get(
                String userId, String sessionId, String key, Class<T> stateType) {
            return super.get(userId, sessionId, key, stateType)
                    .map(state -> stateType.cast(copy(state)));
        }

        private void armAcknowledgementGate() {
            gateArmed.set(true);
        }

        private boolean awaitAcknowledgementSave(Duration timeout) throws InterruptedException {
            return acknowledgementSaveEntered.await(timeout.toMillis(), TimeUnit.MILLISECONDS);
        }

        private void releaseAcknowledgementSave() {
            releaseAcknowledgementSave.countDown();
        }

        private void awaitRelease() {
            try {
                if (!releaseAcknowledgementSave.await(5, TimeUnit.SECONDS)) {
                    throw new IllegalStateException(
                            "timed out waiting to release AgentScope recovery acknowledgement");
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(
                        "AgentScope recovery acknowledgement wait was interrupted", interrupted);
            }
        }

        private static State copy(State state) {
            if (state instanceof AgentState agentState) {
                return AgentState.fromJsonString(agentState.toJson());
            }
            return state;
        }
    }
}
