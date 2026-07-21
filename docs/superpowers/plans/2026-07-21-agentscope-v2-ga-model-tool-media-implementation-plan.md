# AgentScope V2 GA Model, Tool, and Media Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the AgentScope Java 2.0.0 GA model, usage, media, ToolBase, platform sub-agent, recoverable waiting-state, and real-provider verification slice on top of the already-implemented Durable Runtime.

**Architecture:** Durable Runtime remains the source of truth for run/event/model-call rows, owner leases, WAITING CAS operations, terminal state, and kernel snapshots. This slice wraps every provider model in a managed reactive lifecycle, records one durable usage row per logical model call, maps controlled media to V2 content blocks, and runs all platform tools through a guarded ToolBase boundary. Confirmation and external execution are resumed from persisted identities and the original kernel snapshot; no client-supplied tool schema or input is trusted.

**Tech Stack:** Java 21, Spring Boot, Project Reactor, AgentScope Java 2.0.0 GA, MyBatis ports supplied by Durable Runtime, OkHttp/MockWebServer, JUnit 5, Mockito, Reactor Test, Maven Failsafe.

## Global Constraints

- All `io.agentscope` artifacts are exactly `2.0.0`; do not add V1, RC, starter, or session-mysql dependencies.
- Production code must not call `.block()`, `.toIterable()`, `Thread.sleep()`, or use ThreadLocal.
- `ChatModelBase#doStream` has the exact signature `protected Flux<ChatResponse> doStream(List<Msg>, List<ToolSchema>, GenerateOptions)` and reads `RuntimeContext` with `Flux.deferContextual` and `AgentBase.RUNTIME_CONTEXT_KEY`.
- Provider connection creation and blocking stream reads run only on the bounded `agent-model-blocking` scheduler; legacy synchronous tools run only on `agent-tool-blocking`.
- Provider streams/clients close on complete, error, and cancel; disposing a subscription must interrupt a blocking read.
- Every model call receives a distinct durable `modelCallId`; settlement idempotency is exactly `runId:modelCallId`.
- Media accepts only controlled `/media` assets, public HTTP(S), validated data URIs, or explicit Base64; reject `file://`, traversal, private/link-local destinations, unsafe redirects, MIME mismatch, timeout, and size overflow.
- Never persist or log API keys, proxy passwords, Authorization, raw Base64, binary bodies, or signed media query parameters.
- Every registered tool has an object schema with `additionalProperties:false`, explicit `readOnly`, explicit `concurrencySafe`, and a concrete `callAsync(ToolCallParam)` override.
- Tools obtain all request data from `ToolCallParam#getRuntimeContext()`; constructors must not capture request-scoped mutable context.
- Tool and sub-agent deadlines may only shorten `AgentRunContext.deadline`; cancellation and owner fencing are checked before execution and before success/result backfill.
- `REQUIRE_USER_CONFIRM` is recorded as a raw-only candidate, then exposed only after the matching natural `AgentResult` and successful AgentState save.
- `TOOL_SUSPENDED` becomes `WAITING_EXTERNAL` only after natural completion and successful AgentState save.
- WAITING states release Provider, Harness lease, and owner lease while retaining the Redis AgentState session.
- Resume uses the same `sessionId`, `kernelFingerprint`, and `agentDefinitionSnapshotJson`; unavailable snapshots fail closed with `RUN_CONFIG_UNAVAILABLE`.
- This plan must not create or modify Migration, entity, mapper, or repository files.
- Smoke profiles fail fast when required credentials are absent; missing credentials are reported as unverified, never as passed.
- Provider/media capability behavior is implemented only after recording the vendor's official documentation URL, verification date `2026-07-21`, and exact model/version in `docs/agentscope-v2/evidence/2026-07-21-provider-capability-sources.md`; model-name inference and third-party capability tables are forbidden.
- `AgentScopeModelFactory` is the only Spring bean implementing `AgentKernelModelFactory`; no second model factory, model-id-only cache, or Provider-owned singleton model is allowed.
- `PlatformToolRegistry` is part of Kernel construction, not an adapter-only catalog. `toolManifestVersion` and the normalized whitelist fingerprint are fields of `AgentKernelKey`, and every registered `AgentTool` is owned and closed by `AgentKernelResource`.
- Durable Runtime owns `ModelUsageSettlementPort`, `AgentKernelSnapshotBuilder/Resolver`, `PlatformSubAgentRunPort`, `StartAgentExecutionCommand`, `ResumeAgentExecutionCommand`, `RunExecutionSupervisor`, and `CancellationCoordinator`; this plan consumes those exact types and does not create shadow copies.
- Every focused Surefire command uses `-Dsurefire.failIfNoSpecifiedTests=true`; every focused Failsafe command uses `-Dfailsafe.failIfNoSpecifiedTests=true`, so a misspelled or absent test class fails the gate.

## Frozen Durable Runtime Interfaces

This plan consumes these interfaces without renaming or reimplementing their persistence:

```java
AgentModelCallUsageRepository.startCall(String runId, String modelCallId, String provider, String modelCode);
AgentModelCallUsageRepository.completeCall(String runId, String modelCallId, NormalizedModelUsage usage);
AgentModelCallUsageRepository.failCall(String runId, String modelCallId, AgentModelCallStatus status);
AgentModelCallUsageRepository.claimSettlementBatch(String claimOwner, Duration claimLease, int limit);
AgentModelCallUsageRepository.markSettled(long usageId, String claimOwner, String downstreamSettlementId);
AgentModelCallUsageRepository.releaseSettlementForRetry(long usageId, String claimOwner, Instant nextAttemptAt, String sanitizedError);
AgentModelCallUsageRepository.markRunUsageSettledIfAllCallsSettled(String runId);
ModelUsageSettlementPort.settle(String idempotencyKey, NormalizedModelUsage usage); // Mono<String>
RunLeaseGuard.assertLease(String runId, String ownerInstanceId, long ownerEpoch); // Mono<Void>
```

`ModelUsageSettlementPort` and the single audit-ledger adapter bean are produced by Durable Runtime. This plan owns only `ModelCallUsageSettlementWorker`, imports `com.stonewu.fusion.service.ai.run.ModelUsageSettlementPort`, and must not add a second `@ConditionalOnMissingBean` fallback.

```java
AgentWaitingStatePort.recordConfirmationCandidate(String runId, PendingConfirmation candidate);
AgentWaitingStatePort.enterWaitingConfirmation(String runId, long expectedOwnerEpoch, WaitingCheckpoint checkpoint);
AgentWaitingStatePort.enterWaitingExternal(String runId, long expectedOwnerEpoch, WaitingCheckpoint checkpoint, PendingExternalExecution pending);
AgentWaitingStatePort.getPendingConfirmationAuthorized(String runId, long currentUserId, String replyId);
AgentWaitingStatePort.getPendingExternalAuthorized(String runId, long currentUserId, String toolCallId);
AgentWaitingStatePort.resumeConfirmation(ResumeConfirmationCommand command);
AgentWaitingStatePort.resumeExternal(ResumeExternalCommand command);
```

`AgentRunContext` is exactly `(runId, ownerInstanceId, ownerEpoch, deadline)`. `PendingConfirmation`, `PendingExternalExecution`, `WaitingCheckpoint`, both resume commands, and `ResumedAgentRun` retain the exact fields frozen by Durable Runtime.

The following Durable/Kernel signatures are frozen for this plan. Implementers must update the owning dependency/durable task rather than introduce overloads or same-name types in another package:

```java
public interface AgentKernelModelFactory {
    OwnedChatModel create(AgentKernelSpec spec);
}

public interface RunExecutionSupervisor {
    Mono<Void> start(StartAgentExecutionCommand command);
    Mono<Void> resume(ResumeAgentExecutionCommand command);
    Mono<Boolean> interruptOwned(String runId, String ownerInstanceId,
        long ownerEpoch, ExecutionStopReason reason);
    Mono<Void> shutdown(Duration timeout);
}

public interface CancellationCoordinator {
    Mono<AgentRunStatus> cancel(String runId, long currentUserId);
}

public record PlatformSubAgentCommand(
    String parentRunId,
    String parentOwnerInstanceId,
    long parentOwnerEpoch,
    String parentToolCallId,
    String agentName,
    AgentKernelSpec kernelSpec,
    List<Msg> messages,
    ProjectContext projectContext,
    Instant deadline) {}

public record PlatformSubAgentRun(
    String childRunId,
    String parentRunId,
    String parentToolCallId,
    String agentName,
    AgentRunStatus status) {}

public interface PlatformSubAgentRunPort {
    Mono<PlatformSubAgentRun> start(PlatformSubAgentCommand command);
    Mono<Void> cancelChildren(String parentRunId);
}
```

---

### Task 1: Durable Model Call Usage Ledger

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model/ModelCallTicket.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model/ModelCallUsageLedger.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model/DefaultModelCallUsageLedger.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/model/DefaultModelCallUsageLedgerTests.java`

**Interfaces:** Consumes the frozen repository and `AgentRuntimeSchedulers.journal()`. Produces `Mono<ModelCallTicket> start(AgentRunContext run,String provider,String modelCode)`, `Mono<Void> complete(ModelCallTicket ticket,NormalizedModelUsage usage)`, `Mono<Void> fail(ModelCallTicket ticket,Throwable failure)`, and `Mono<Void> cancel(ModelCallTicket ticket)`.

- [ ] **Step 1: Write the failing call-identity and terminal-status tests**

```java
@Test void oneRunCreatesDistinctDurableCallIds() {
    StepVerifier.create(Flux.concat(ledger.start(run, "openai", "gpt-4.1"), ledger.start(run, "openai", "gpt-4.1")))
        .recordWith(ArrayList::new).expectNextCount(2)
        .consumeRecordedWith(rows -> assertNotEquals(rows.get(0).modelCallId(), rows.get(1).modelCallId()))
        .verifyComplete();
}
@Test void cancelWritesCancelledInsteadOfFailed() {
    ModelCallTicket ticket = new ModelCallTicket("run-1", "call-1", "openai", "gpt-4.1");
    StepVerifier.create(ledger.cancel(ticket)).verifyComplete();
    verify(repository).failCall("run-1", "call-1", AgentModelCallStatus.CANCELLED);
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `cd D:\develop\my\ai-fusion-video\ai-fusion-video; .\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true -Dtest=DefaultModelCallUsageLedgerTests test`
Expected: FAIL because `ModelCallUsageLedger` and `ModelCallTicket` do not exist.

- [ ] **Step 3: Implement the ledger on the journal scheduler**

```java
public record ModelCallTicket(String runId, String modelCallId, String provider, String modelCode) {}

public interface ModelCallUsageLedger {
    Mono<ModelCallTicket> start(AgentRunContext run, String provider, String modelCode);
    Mono<Void> complete(ModelCallTicket ticket, NormalizedModelUsage usage);
    Mono<Void> fail(ModelCallTicket ticket, Throwable failure);
    Mono<Void> cancel(ModelCallTicket ticket);
}

public Mono<ModelCallTicket> start(AgentRunContext run, String provider, String modelCode) {
    return Mono.fromCallable(() -> {
        String id = UUID.randomUUID().toString();
        repository.startCall(run.runId(), id, provider, modelCode);
        return new ModelCallTicket(run.runId(), id, provider, modelCode);
    }).subscribeOn(schedulers.journal());
}
```

- [ ] **Step 4: Run the focused tests**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true -Dtest=DefaultModelCallUsageLedgerTests test`
Expected: PASS; repository calls occur on an `agent-journal-*` thread.

- [ ] **Step 5: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/model/DefaultModelCallUsageLedgerTests.java
git commit -m "feat: add durable model call usage ledger"
```

### Task 2: Idempotent Usage Settlement Worker

**Files:**
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/ModelUsageSettlementPort.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AuditLedgerModelUsageSettlementAdapter.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model/ModelCallUsageSettlementWorker.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model/SettlementBackoff.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/model/ModelCallUsageSettlementWorkerTests.java`

**Interfaces:** Consumes `claimSettlementBatch`, `ModelUsageSettlementPort`, both claim completion methods, and `markRunUsageSettledIfAllCallsSettled`. Produces `Mono<Integer> settleBatch(String owner)`.

- [ ] **Step 1: Write failing success, retry, and idempotency-key tests**

```java
@Test void settlesOutsideClaimAndUsesCallKey() {
    when(repository.claimSettlementBatch("node-a", Duration.ofSeconds(20), 50)).thenReturn(List.of(usage));
    when(port.settle("run-1:call-7", usage.normalizedUsage())).thenReturn(Mono.just("bill-9"));
    StepVerifier.create(worker.settleBatch("node-a")).expectNext(1).verifyComplete();
    verify(repository).markSettled(usage.getId(), "node-a", "bill-9");
    verify(repository).markRunUsageSettledIfAllCallsSettled("run-1");
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true -Dtest=ModelCallUsageSettlementWorkerTests test`
Expected: FAIL to compile because the worker is absent; the Durable Runtime-owned Port and adapter must already compile and the command must not report `No tests to run`.

- [ ] **Step 3: Implement ordered claim, external settlement, acknowledgement, and retry**

```java
public Mono<Integer> settleBatch(String owner) {
    return Mono.fromCallable(() -> repository.claimSettlementBatch(owner, claimLease, batchSize))
        .subscribeOn(schedulers.journal())
        .flatMapMany(Flux::fromIterable)
        .concatMap(row -> settleOne(owner, row))
        .reduce(0, Integer::sum);
}

private Mono<Integer> settleOne(String owner, AgentModelCallUsage row) {
    String key = row.getRunId() + ":" + row.getModelCallId();
    return settlementPort.settle(key, row.normalizedUsage())
        .flatMap(receipt -> journal(() -> repository.markSettled(row.getId(), owner, receipt)))
        .flatMap(marked -> marked
            ? journal(() -> repository.markRunUsageSettledIfAllCallsSettled(row.getRunId())).thenReturn(1)
            : Mono.error(new SettlementClaimLostException(row.getId(), owner)))
        .onErrorResume(error -> journal(() -> repository.releaseSettlementForRetry(
                row.getId(), owner, backoff.next(row.getSettlementAttempts()), sanitizer.message(error)))
            .thenReturn(0));
}
```

- [ ] **Step 4: Run the focused tests**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=ModelCallUsageSettlementWorkerTests,AuditLedgerModelUsageSettlementAdapterTests,AgentUsageSettlementConfigurationTests" test`
Expected: PASS; one successful settlement returns `1`, a retry release returns `0`, stale claims cannot mark a run settled, and exactly one Durable Runtime audit-ledger adapter bean is bound.

- [ ] **Step 5: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model/ModelCallUsageSettlementWorker.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model/SettlementBackoff.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/model/ModelCallUsageSettlementWorkerTests.java
git commit -m "feat: settle model usage with call-level idempotency"
```

### Task 3: Managed ChatModel Lifecycle and Runtime Guard

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model/ManagedAgentScopeChatModel.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model/ModelUsageAccumulator.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/model/ManagedAgentScopeChatModelTests.java`

**Interfaces:** Consumes `ModelCallUsageLedger` and `AgentRunContext`. `ManagedAgentScopeChatModel extends ChatModelBase implements AutoCloseable`, wraps a provider `ChatModelBase`, and closes its owned delegate resources idempotently; the Kernel plan's outer `StateStoreGuardedChatModel` remains the single Provider pre-call StateStore guard.

- [ ] **Step 1: Write failing missing-context and three-terminal-path tests**

```java
@Test void missingRuntimeContextFailsBeforeDelegateSubscription() {
    StepVerifier.create(managed.stream(messages, tools, options)).expectErrorMatches(e -> e.getMessage().contains("RuntimeContext")).verify();
    assertEquals(0, delegateSubscriptions.get());
}
@Test void cancellationClosesDelegateAndCancelsLedger() {
    StepVerifier.create(withRuntime(managed.stream(messages, tools, options))).thenCancel().verify();
    verify(ledger).cancel(any(ModelCallTicket.class));
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true -Dtest=ManagedAgentScopeChatModelTests test`
Expected: FAIL because the managed wrapper is absent.

- [ ] **Step 3: Implement `deferContextual` plus `usingWhen` lifecycle**

```java
protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
    return Flux.deferContextual(view -> {
        RuntimeContext runtime = view.getOrDefault(AgentBase.RUNTIME_CONTEXT_KEY, null);
        if (runtime == null) return Flux.error(new IllegalStateException("AgentScope RuntimeContext missing"));
        AgentRunContext run = Objects.requireNonNull(runtime.get(AgentRunContext.class), "AgentRunContext missing");
        ModelUsageAccumulator usage = new ModelUsageAccumulator();
        return Flux.usingWhen(
            ledger.start(run, provider, getModelName()),
            ticket -> delegate.stream(messages, tools, options).doOnNext(usage::accept),
            ticket -> ledger.complete(ticket, usage.snapshot()),
            (ticket, error) -> ledger.fail(ticket, error),
            ledger::cancel);
    });
}
```

- [ ] **Step 4: Run the focused tests**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true -Dtest=ManagedAgentScopeChatModelTests test`
Expected: PASS for complete/error/cancel, distinct subscriptions, missing context, and double close.

- [ ] **Step 5: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/model ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/model/ManagedAgentScopeChatModelTests.java
git commit -m "feat: manage provider streams and usage lifecycle"
```

### Task 4: Five Official GA Provider Builders

**Files:**
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AiProviderService.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeModelFactory.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/OpenAiCompatibleAiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/AnthropicAiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/GeminiAiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/DashScopeAiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/OllamaAiProvider.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/VertexAiProvider.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/AgentScopeGaProviderFactoryTests.java`

**Interfaces:** `AiProvider#createAgentScopeModel(AiProviderContext)` returns the official/custom `ChatModelBase`; the unique Spring bean `AgentScopeModelFactory implements AgentKernelModelFactory` implements `OwnedChatModel create(AgentKernelSpec spec)`. It consumes the immutable `spec.model()`, wraps the Provider model in `ManagedAgentScopeChatModel`, and never caches solely by model ID.

- [ ] **Step 1: Write failing exact-class and ownership tests**

```java
@ParameterizedTest @MethodSource("providers")
void usesOfficialGaProviderClass(AiProvider provider, AiProviderContext context, String className) {
    ChatModelBase delegate = provider.createAgentScopeModel(context);
    assertEquals(className, delegate.getClass().getName());
    closeIfCloseable(delegate);
}
@Test void modelFactoryDoesNotReuseDifferentConfigFingerprints() {
    assertNotSame(factory.create(specV1).model(), factory.create(specV2).model());
}
@Test void applicationHasExactlyOneKernelModelFactory() {
    assertThat(context.getBeansOfType(AgentKernelModelFactory.class))
        .containsOnlyKeys("agentScopeModelFactory");
    assertThat(context.getBean(AgentKernelModelFactory.class))
        .isInstanceOf(AgentScopeModelFactory.class);
}
@Test void vertexImplementsTheNarrowChatModelBaseContract() {
    assertThat(vertex.createAgentScopeModel(vertexContext)).isInstanceOf(ChatModelBase.class);
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true -Dtest=AgentScopeGaProviderFactoryTests test`
Expected: FAIL because the Kernel factory still returns an unmanaged Provider model, not every Provider including Vertex implements the narrow `ChatModelBase` contract, or two config snapshots are not independently owned.

- [ ] **Step 3: Replace the provider contract and wrap official models**

```java
public interface AiProvider {
    boolean supports(String platform);
    ChatModel createChatModel(AiProviderContext context);
    ChatModelBase createAgentScopeModel(AiProviderContext context);
    List<RemoteModelVO> listRemoteModels(AiProviderContext context);
}

@Component("agentScopeModelFactory")
public final class AgentScopeModelFactory implements AgentKernelModelFactory {
    private final AiProviderService providerService;
    private final ModelCallUsageLedger usageLedger;

    @Override
    public OwnedChatModel create(AgentKernelSpec spec) {
        AiModel snapshot = Objects.requireNonNull(spec.model(), "Agent model snapshot unavailable");
        ChatModelBase provider = providerService.createAgentScopeModel(snapshot);
        ManagedAgentScopeChatModel managed = new ManagedAgentScopeChatModel(
            providerService.providerCode(snapshot), provider, usageLedger);
        return new OwnedChatModel() {
            private final AtomicBoolean closed = new AtomicBoolean();
            @Override public ChatModelBase model() { return managed; }
            @Override public void close() {
                if (closed.compareAndSet(false, true)) managed.close();
            }
        };
    }
}
```

- [ ] **Step 4: Run provider tests and dependency verification**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true -Dtest=AgentScopeGaProviderFactoryTests test; .\mvnw.cmd dependency:tree -Dincludes=io.agentscope`
Expected: PASS; OpenAI/Anthropic/Gemini/DashScope/Ollama/Vertex compile against `ChatModelBase`, exactly one `AgentKernelModelFactory` bean exists, and the tree contains only `2.0.0` GA modules.

- [ ] **Step 5: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeModelFactory.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/AgentScopeGaProviderFactoryTests.java
git commit -m "feat: build five official AgentScope GA providers"
```

### Task 5: OpenAI Responses and Necessary Custom Provider Contracts

**Files:**
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider/OpenAiResponsesAgentScopeModel.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/OpenAiResponsesAgentScopeModelTests.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/AnthropicAiProviderTests.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/GeminiAiProviderTests.java`
- Modify: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/VertexAiProviderTests.java`

**Interfaces:** Responses extends `ChatModelBase`; all custom blocking reads use `AgentRuntimeSchedulers.modelBlocking()` and expose idempotent close. The dependency/kernel plan has already removed obsolete proxy/formatter shims in its atomic GA compile cut; this task proves official Anthropic/Gemini/Vertex equivalence remains intact after managed lifecycle wrapping.

- [ ] **Step 1: Write failing stream parsing and close tests**

```java
@Test void responsesClosesOnCompleteErrorAndCancel() {
    assertAll(() -> verifyTerminal(COMPLETE), () -> verifyTerminal(ERROR), () -> verifyTerminal(CANCEL));
    assertEquals(3, fakeStreams.closedCount());
}
@Test void providerCreationOccursOnModelScheduler() {
    StepVerifier.create(withRuntime(model.stream(messages, tools, options))).expectNextCount(1).verifyComplete();
    assertTrue(fakeStreams.openThread().startsWith("agent-model-blocking-"));
}
@Test void gaFormatterPreservesToolResultOrdering() {
    ChatRequest request = captureGeminiRequest(toolUseThenResultMessages());
    assertThat(request.contents()).extracting(Content::getRole)
        .containsExactly("model", "user");
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=OpenAiResponsesAgentScopeModelTests,AnthropicAiProviderTests,GeminiAiProviderTests,VertexAiProviderTests" test`
Expected: FAIL because Responses still implements the old `Model`, blocking reads remain unmanaged, or a GA-equivalence regression is exposed.

- [ ] **Step 3: Implement the GA `doStream` resource boundary**

```java
@Override
protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
    return Flux.deferContextual(view -> {
        RuntimeContext runtime = view.getOrDefault(AgentBase.RUNTIME_CONTEXT_KEY, null);
        if (runtime == null || runtime.get(AgentRunContext.class) == null) {
            return Flux.error(new IllegalStateException("AgentScope RuntimeContext missing"));
        }
        AgentRunContext run = runtime.get(AgentRunContext.class);
        return Flux.using(
                () -> responseStreamFactory.open(buildRequestParams(messages, tools, options, run.deadline())),
                handle -> handle.events().map(this::mapEvent),
                ResponseStreamHandle::close)
            .subscribeOn(schedulers.modelBlocking());
    });
}

@Override public void close() {
    if (closed.compareAndSet(false, true)) activeHandles.forEach(ResponseStreamHandle::close);
}
```

- [ ] **Step 4: Reassert proxy/formatter behavior on official GA builders**

```java
AnthropicChatModel.builder()
    .apiKey(context.getApiKey())
    .modelName(context.getModelName())
    .proxy(proxyConfig(context))
    .build();

GeminiChatModel.builder()
    .apiKey(context.getApiKey())
    .modelName(context.getModelName())
    .proxy(proxyConfig(context))
    .build();
```

Run: `rg -n "AnthropicAgentScopeProxySupport|ProxyAwareAnthropicChatModel|GeminiToolResponseAwareChatFormatter|VertexAgentScopeProxySupport" ai-fusion-video/src/main ai-fusion-video/src/test`

Expected: no production or test reference remains; official builder tests are the sole compatibility proof.

- [ ] **Step 5: Run custom-provider tests**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=OpenAiResponsesAgentScopeModelTests,AnthropicAiProviderTests,GeminiAiProviderTests,VertexAiProviderTests" test`
Expected: PASS; Responses lifecycle and official GA proxy/formatter/Vertex behavior are contract-tested, obsolete class names have no production references, and no global `boundedElastic` remains in custom providers.

- [ ] **Step 6: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider
git commit -m "feat: migrate required custom provider contracts to GA"
```

### Task 6: Safe Media Resolution and Transport Policy

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AiMediaInputVO.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AiChatReqVO.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/media/AgentMediaTransportPolicy.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/media/ResolvedAgentMedia.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/media/SafeAgentMediaResolver.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/media/SafeAgentMediaResolverTests.java`
- Create: `docs/agentscope-v2/evidence/2026-07-21-provider-capability-sources.md`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/media/ProviderCapabilityEvidenceTests.java`

**Interfaces:** Produces `Mono<ResolvedAgentMedia> resolve(AiMediaInputVO, AgentMediaTransportPolicy)`; policy is explicit model configuration, never inferred from a model name or failed retry. Evidence rows are `(provider, modelCode, modelVersion, capability, officialUrl, verifiedOn)` and reject non-vendor hosts or a date other than `2026-07-21`.

- [ ] **Step 1: Enumerate the exact Provider/model codes used by the adapters and smoke environment, verify each capability against vendor documentation, and write the evidence contract test**

Run: `rg -n "modelName\(|modelCode|AFV_SMOKE_.*_MODEL|referenceImage|referenceVideo" ai-fusion-video/src/main ai-fusion-video/src/test ai-fusion-video/src/main/resources/model-presets`

Expected: a concrete list of model codes and media behaviors to verify; no evidence row may use a model-family nickname in place of the actual configured code.

```java
@Test void capabilityEvidenceUsesOnlyOfficialVendorSources() {
    List<CapabilityEvidence> rows = parser.read("../docs/agentscope-v2/evidence/2026-07-21-provider-capability-sources.md");
    assertThat(rows).isNotEmpty().allSatisfy(row -> {
        assertThat(row.verifiedOn()).isEqualTo(LocalDate.of(2026, 7, 21));
        assertThat(row.modelCode()).isNotBlank();
        assertThat(row.modelVersion()).isNotBlank();
        assertThat(officialHosts.get(row.provider())).contains(URI.create(row.officialUrl()).getHost());
    });
}
```

- [ ] **Step 2: Write failing URL/Base64/SSRF/MIME tests**

```java
@Test void rejectsFileAndPrivateDestinations() {
    StepVerifier.create(resolve("file:///etc/passwd")).expectErrorCode("MEDIA_SOURCE_FORBIDDEN").verify();
    StepVerifier.create(resolve("http://127.0.0.1/a.png")).expectErrorCode("MEDIA_SSRF_BLOCKED").verify();
}
@Test void validatesEveryRedirectAndSize() {
    server.enqueue(redirect("http://169.254.169.254/latest/meta-data"));
    StepVerifier.create(resolve(server.url("/a").toString())).expectErrorCode("MEDIA_SSRF_BLOCKED").verify();
}
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=ProviderCapabilityEvidenceTests,SafeAgentMediaResolverTests" test`
Expected: FAIL because the evidence document, resolver, and media VO do not exist.

- [ ] **Step 4: Implement explicit transport selection and guarded download**

```java
public Mono<ResolvedAgentMedia> resolve(AiMediaInputVO input, AgentMediaTransportPolicy policy) {
    return Mono.defer(() -> {
        URI uri = URI.create(input.getSource());
        if ("file".equalsIgnoreCase(uri.getScheme())) return Mono.error(mediaError("MEDIA_SOURCE_FORBIDDEN"));
        if ("data".equalsIgnoreCase(uri.getScheme())) return Mono.just(parseDataUri(input, policy));
        if (input.getSource().startsWith("/media/")) return resolveControlledLocal(input, policy);
        if (!Set.of("http", "https").contains(uri.getScheme())) return Mono.error(mediaError("MEDIA_SOURCE_FORBIDDEN"));
        return policy.urlAllowed() ? validatePublicUrl(uri, input, policy) : downloadAsBase64(uri, input, policy);
    }).subscribeOn(schedulers.modelBlocking());
}
```

- [ ] **Step 5: Run media security and evidence tests**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=ProviderCapabilityEvidenceTests,SafeAgentMediaResolverTests" test`
Expected: PASS for official evidence coverage, `/media`, public URL, data URI, Base64, redirect revalidation, private IP, timeout, maximum bytes, and MIME mismatch.

- [ ] **Step 6: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AiMediaInputVO.java ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AiChatReqVO.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/media ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/media docs/agentscope-v2/evidence/2026-07-21-provider-capability-sources.md
git commit -m "feat: add secure AgentScope media resolution"
```

### Task 7: V2 Media ContentBlock Mapping

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/media/AgentScopeMediaBlockMapper.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/message/AgentScopeMessageMapper.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/message/AgentScopeMessageMapperTests.java`

**Interfaces:** Produces `Mono<UserMessage> toUserMessage(String text, List<AiMediaInputVO> media, AgentMediaTransportPolicy policy)` while preserving content order.

- [ ] **Step 1: Write failing typed-block and redaction tests**

```java
@Test void preservesTextImageVideoOrderAndSourceType() {
    StepVerifier.create(mapper.toUserMessage("describe", inputs, policy)).assertNext(message -> {
        assertInstanceOf(TextBlock.class, message.getContent().get(0));
        assertInstanceOf(ImageBlock.class, message.getContent().get(1));
        assertInstanceOf(VideoBlock.class, message.getContent().get(2));
    }).verifyComplete();
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true -Dtest=AgentScopeMessageMapperTests test`
Expected: FAIL because only the old pure-text mapper exists.

- [ ] **Step 3: Map exact V2 source constructors without setting role**

```java
private ContentBlock mediaBlock(ResolvedAgentMedia media) {
    MediaSource source = media.url() != null
        ? new URLSource(media.url(), media.mimeType())
        : new Base64Source(media.mimeType(), media.base64Data());
    return switch (media.kind()) {
        case IMAGE -> ImageBlock.builder().source(source).build();
        case VIDEO -> VideoBlock.builder().source(source).build();
        case AUDIO -> AudioBlock.builder().source(source).build();
    };
}
```

- [ ] **Step 4: Run typed-message tests**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true -Dtest=AgentScopeMessageMapperTests test`
Expected: PASS; no call to the unsupported `UserMessage.Builder.role` setter, and captured DB/log payloads contain hashes/references rather than Base64.

- [ ] **Step 5: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/message/AgentScopeMessageMapper.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/media/AgentScopeMediaBlockMapper.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/message/AgentScopeMessageMapperTests.java
git commit -m "feat: preserve typed media in AgentScope messages"
```

### Task 8: ToolBase Contract, Manifest, and Whitelist

**Files:**
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/AbstractPlatformAgentTool.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformToolDescriptor.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformToolManifest.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformToolRegistry.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/OwnedPlatformToolkit.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelToolRegistry.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelToolkitResources.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelSpec.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentScopeHarnessFactory.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelResource.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformToolRegistryTests.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentScopeToolHarnessIntegrationTests.java`

**Interfaces:** `PlatformToolRegistry implements AgentKernelToolRegistry` and produces `AgentKernelToolkitResources register(AgentKernelSpec spec, Toolkit toolkit)`. It validates the frozen `AgentKernelSpec.toolManifest()/toolWhitelist()/toolWhitelistVersion()` against concrete tools, registers the exact intersection into the Harness-owned Toolkit, and returns all tool closeables to `AgentKernelResource`. The dependency plan already puts canonical `toolManifestFingerprint` and `toolWhitelistVersion` into `AgentKernelKey`; this task must not create a second key or spec.

- [ ] **Step 1: Write failing reflection, schema, and whitelist tests**

```java
@Test void everyRegisteredToolOverridesCallAsyncAndHasClosedSchema() throws Exception {
    for (AgentTool tool : registry.toolsFor("script_assistant")) {
        assertNotNull(tool.getClass().getDeclaredMethod("callAsync", ToolCallParam.class));
        assertEquals("object", tool.getInputSchema().get("type"));
        assertEquals(false, tool.getInputSchema().get("additionalProperties"));
    }
}
@Test void registryRegistersOnlyThePersistedManifestAndWhitelist() {
    Toolkit toolkit = new Toolkit();
    AgentKernelToolkitResources resources = registry.register(specWithEchoWhitelist, toolkit);
    assertThat(toolkit.getToolNames()).containsExactly("echo");
    resources.close();
    assertThat(echoTool.closeCount()).hasValue(1);
}
@Test void realHarnessInvokesWhitelistedToolAndClosesItOnce() {
    StepVerifier.create(invoker.call(specWithEchoTool, runtimeContext, userMessage))
        .assertNext(message -> assertThat(message.getTextContent()).contains("echo-result"))
        .verifyComplete();
    cache.invalidate(specWithEchoTool.key());
    assertThat(echoTool.closeCount()).hasValue(1);
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=PlatformToolRegistryTests,AgentScopeToolHarnessIntegrationTests" test`
Expected: FAIL because tools implement the old interface and the conditional no-op `AgentKernelToolRegistry` leaves the real Harness without the approved platform tool.

- [ ] **Step 3: Add the abstract override and explicit manifest**

```java
public abstract class AbstractPlatformAgentTool extends ToolBase {
    protected AbstractPlatformAgentTool(Builder builder) { super(builder); }
    @Override public abstract Mono<ToolResultBlock> callAsync(ToolCallParam param);
}

public record PlatformToolDescriptor(String name, boolean readOnly, boolean concurrencySafe) {}

public record OwnedPlatformToolkit(List<AutoCloseable> resources)
        implements AgentKernelToolkitResources {
    @Override public void close() {
        RuntimeException failure = null;
        for (AutoCloseable resource : resources) {
            try { resource.close(); }
            catch (Exception error) {
                if (failure == null) failure = new RuntimeException("toolkit close failed", error);
                else failure.addSuppressed(error);
            }
        }
        if (failure != null) throw failure;
    }
}

public static final String WHITELIST_VERSION = "afv-tools-v1";

private static PlatformToolDescriptor readOnly(String name) {
    return new PlatformToolDescriptor(name, true, true);
}
private static PlatformToolDescriptor mutating(String name) {
    return new PlatformToolDescriptor(name, false, false);
}

private static final Map<String, PlatformToolDescriptor> MANIFEST = Stream.of(
    readOnly("get_asset"),
    readOnly("get_generation_model_capabilities"),
    readOnly("get_project"),
    readOnly("get_project_script"),
    readOnly("get_script"),
    readOnly("get_script_episode"),
    readOnly("get_script_scene"),
    readOnly("get_script_structure"),
    readOnly("get_storyboard"),
    readOnly("get_storyboard_scene_items"),
    readOnly("list_my_projects"),
    readOnly("list_project_assets"),
    readOnly("list_project_storyboards"),
    readOnly("query_asset_items"),
    readOnly("query_asset_metadata"),
    mutating("add_asset_item"),
    mutating("batch_create_asset_items"),
    mutating("batch_create_assets"),
    mutating("create_asset"),
    mutating("generate_image"),
    mutating("generate_video"),
    mutating("insert_storyboard_item"),
    mutating("manage_script_scenes"),
    mutating("save_script_episode"),
    mutating("save_script_scene_items"),
    mutating("save_storyboard_episode"),
    mutating("save_storyboard_scene_shots"),
    mutating("update_asset"),
    mutating("update_asset_image"),
    mutating("update_script"),
    mutating("update_script_info"),
    mutating("update_script_scene"),
    mutating("update_storyboard_item_frame"),
    mutating("update_storyboard_item_video")
).collect(Collectors.toUnmodifiableMap(PlatformToolDescriptor::name, Function.identity()));
```

- [ ] **Step 4: Connect the governed Toolkit to the production Kernel**

```java
@Component
@RequiredArgsConstructor
public final class PlatformToolRegistry implements AgentKernelToolRegistry {
    private final PlatformToolManifest manifest;
    private final List<AgentTool> tools;

@Override
public AgentKernelToolkitResources register(AgentKernelSpec spec, Toolkit toolkit) {
    if (!PlatformToolManifest.WHITELIST_VERSION.equals(spec.toolWhitelistVersion())) {
        throw new BusinessException("RUN_CONFIG_UNAVAILABLE", "工具 manifest 版本不可用");
    }
    List<AgentTool> tools = toolsFor(spec.agentDefinitionStableKey()).stream()
        .filter(tool -> spec.toolWhitelist().contains(tool.getName()))
        .sorted(Comparator.comparing(AgentTool::getName))
        .toList();
    if (tools.size() != spec.toolWhitelist().size()
            || !manifestOf(tools).equals(spec.toolManifest())) {
        throw new BusinessException("RUN_CONFIG_UNAVAILABLE", "工具白名单包含不可用工具");
    }
    tools.forEach(toolkit::registerAgentTool);
    List<AutoCloseable> owned = tools.stream()
        .filter(AutoCloseable.class::isInstance)
        .map(AutoCloseable.class::cast)
        .toList();
    return new OwnedPlatformToolkit(owned);
}
}
```

`manifestOf` emits `AgentKernelToolManifest` rows sorted by tool name using the concrete schema SHA-256, `readOnly`, and `concurrencySafe` metadata. The already-frozen `AgentScopeHarnessFactory` calls this registry exactly once; `AgentKernelResource.close()` closes the returned resources exactly once without closing the shared StateStore.

- [ ] **Step 5: Run registry and real Harness tests**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=PlatformToolRegistryTests,AgentScopeToolHarnessIntegrationTests,AgentScopeHarnessFactoryTests,AgentKernelResourceTests" test`
Expected: PASS only when every executor is in the manifest, the whitelist is enforced, Kernel keys change with manifest/whitelist, a real Harness invokes the tool, and eviction closes the tool exactly once.

- [ ] **Step 6: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformToolRegistryTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentScopeToolHarnessIntegrationTests.java
git commit -m "feat: wire governed tools into AgentScope kernels"
```

### Task 9: Tool Blocking, Deadline, Cancellation, and Fence Boundary

**Files:**
- Replace: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeToolAdapter.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/ToolExecutionContext.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/ToolExecutionHandle.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/ToolCancellationReason.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/ToolExecutionRequest.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/InProcessToolExecutionHandle.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/CancellableToolExecutor.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/CancellableToolExecutionPort.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/DefaultCancellableToolExecutionPort.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/tool/GenerateImageToolExecutor.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/tool/GenerateVideoToolExecutor.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/generation/consumer/ImageGenerationConsumer.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/generation/consumer/VideoGenerationConsumer.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/generation/ImageGenerationService.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/generation/VideoGenerationService.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeToolAdapterTests.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/generation/consumer/ImageGenerationConsumerTests.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/generation/consumer/VideoGenerationConsumerTests.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentToolCancellationIT.java`

**Interfaces:** Consumes `AgentRunContext`, `CancellationContext`, `StateStoreFailureGuard`, `Mono<Void> RunLeaseGuard.assertLease(String runId,String ownerInstanceId,long ownerEpoch)`, and `AgentRuntimeSchedulers.toolBlocking()`. Produces `Mono<ToolExecutionHandle> CancellableToolExecutionPort.start(ToolExecutionRequest)`; `ToolExecutionHandle` owns one execution and exposes `result()`, `cancel(reason)`, and idempotent `close()`. Image/video consumers expose `awaitTerminal`, `cancelIfActive`, and transactional `completeWithArtifactsIfActive`; cancellation wins the same-row CAS and suppresses artifacts/status success.

- [ ] **Step 1: Write failing scheduler, timeout, cancel, and lost-fence tests**

```java
@Test void lostFenceSuppressesSuccessBackfill() {
    when(leaseGuard.assertLease("run-1", "node-a", 7)).thenReturn(Mono.empty(), Mono.error(new LostOwnerLeaseException()));
    StepVerifier.create(adapter.callAsync(param)).expectError(LostOwnerLeaseException.class).verify();
    verifyNoInteractions(successBackfill);
}
@Test void timeoutCancelsTheDomainHandle() {
    when(port.start(any())).thenReturn(Mono.just(handle));
    when(handle.result()).thenReturn(Mono.never());
    StepVerifier.withVirtualTime(() -> adapter.callAsync(param))
        .thenAwait(Duration.ofSeconds(31))
        .expectError(ToolTimeoutException.class)
        .verify();
    verify(handle).cancel(ToolCancellationReason.TIMEOUT);
    verify(handle).close();
}
@Test void cancelRaceCannotPersistGeneratedArtifact() {
    ToolExecutionHandle handle = imageExecutor.start(request.input(), request.context()).block();
    StepVerifier.create(handle.cancel(ToolCancellationReason.RUN_CANCELLED)).verifyComplete();
    consumer.completeRemoteResult(handle.executionId(), remoteImage);
    ImageTask cancelled = imageGenerationService.getByTaskId(handle.executionId());
    assertThat(cancelled.getStatus()).isEqualTo(3);
    assertThat(imageGenerationService.listItems(cancelled.getId()))
        .allMatch(item -> item.getImageUrl() == null);
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=AgentScopeToolAdapterTests,ImageGenerationConsumerTests,VideoGenerationConsumerTests" test`
Expected: FAIL because the cancellable port/handle does not exist, the old adapter captures request context, and both consumers can still publish success after timeout/cancel.

- [ ] **Step 3: Implement the explicit cancellable execution contract**

```java
public enum ToolCancellationReason { RUN_CANCELLED, TIMEOUT, FENCE_LOST, STREAM_CANCELLED }

public interface ToolExecutionHandle {
    String executionId();
    Mono<String> result();
    Mono<Void> cancel(ToolCancellationReason reason);
    Mono<Void> close();
}

public interface CancellableToolExecutionPort {
    Mono<ToolExecutionHandle> start(ToolExecutionRequest request);
}

public record ToolExecutionRequest(
    String executionId,
    ToolExecutor executor,
    String input,
    ToolExecutionContext context) {}

public interface CancellableToolExecutor extends ToolExecutor {
    Mono<ToolExecutionHandle> start(String input, ToolExecutionContext context);
}

public Mono<ToolExecutionHandle> start(ToolExecutionRequest request) {
    if (request.executor() instanceof CancellableToolExecutor cancellable) {
        return cancellable.start(request.input(), request.context());
    }
    AtomicBoolean cancelled = new AtomicBoolean();
    Mono<String> result = Mono.fromCallable(() -> request.executor().execute(
            request.input(), request.context()))
        .subscribeOn(schedulers.toolBlocking())
        .filter(ignored -> !cancelled.get())
        .switchIfEmpty(Mono.error(new BusinessException("TOOL_CANCELLED", "工具执行已取消")))
        .cache();
    return Mono.just(new InProcessToolExecutionHandle(
        request.executionId(), result, cancelled, schedulers.toolBlocking()));
}
```

- [ ] **Step 4: Use `usingWhen` so every terminal signal cancels or closes the handle**

```java
public Mono<ToolResultBlock> callAsync(ToolCallParam param) {
    ToolRuntime runtime = requireRuntime(param.getRuntimeContext());
    Duration remaining = Duration.between(clock.instant(), runtime.run().deadline());
    return runtime.cancellation().checkpoint()
        .then(Mono.fromRunnable(() -> stateGuard.throwIfFailed(
            new StateStoreSlot(runtime.userId(), runtime.sessionId()))))
        .then(leaseGuard.assertLease(runtime.run().runId(), runtime.run().ownerInstanceId(), runtime.run().ownerEpoch()))
        .then(Mono.usingWhen(
            executionPort.start(new ToolExecutionRequest(
                param.getToolUseBlock().getId(), executor, json(param.getInput()), runtime.toolContext())),
            handle -> handle.result()
                .timeout(remaining, Mono.error(new ToolTimeoutException(executor.getToolName())))
                .flatMap(result -> runtime.cancellation().checkpoint()
                    .then(leaseGuard.assertLease(runtime.run().runId(), runtime.run().ownerInstanceId(), runtime.run().ownerEpoch()))
                    .thenReturn(toolResult(param, result))),
            ToolExecutionHandle::close,
            (handle, failure) -> handle.cancel(reason(failure)).then(handle.close()),
            handle -> handle.cancel(ToolCancellationReason.STREAM_CANCELLED).then(handle.close())));
}

private ToolCancellationReason reason(Throwable failure) {
    if (failure instanceof ToolTimeoutException) return ToolCancellationReason.TIMEOUT;
    if (failure instanceof LostOwnerLeaseException) return ToolCancellationReason.FENCE_LOST;
    return ToolCancellationReason.RUN_CANCELLED;
}
```

- [ ] **Step 5: Make image/video generation cancellation win the persistence race**

```java
public Mono<ImageTask> awaitTerminal(String taskId, Instant deadline) {
    return Flux.interval(Duration.ZERO, Duration.ofSeconds(2))
        .concatMap(tick -> Mono.fromCallable(() -> imageGenerationService.getByTaskId(taskId))
            .subscribeOn(schedulers.toolBlocking()))
        .takeUntil(task -> task.getStatus() == 2 || task.getStatus() == 3)
        .filter(task -> task.getStatus() == 2 || task.getStatus() == 3)
        .next()
        .timeout(Duration.between(clock.instant(), deadline));
}

public Mono<Void> cancelIfActive(String taskId, String reason) {
    return Mono.fromRunnable(() -> imageGenerationService.cancelIfActive(taskId, reason))
        .subscribeOn(schedulers.toolBlocking()).then();
}

private void completeRemoteResult(ImageTask task) {
    boolean completed = imageGenerationService.completeWithArtifactsIfActive(
        task.getTaskId(), () -> persistImageItems(task));
    if (!completed) log.info("[ImageConsumer] suppress cancelled completion: taskId={}", task.getTaskId());
}
```

Implement the same three methods for video with a three-second poll interval. `completeWithArtifactsIfActive` locks the task row, requires status `0` or `1`, persists artifacts, and changes status to `2` in one transaction. `cancelIfActive` changes only `0/1 -> 3`; therefore cancel/complete races have one database winner. Replace both `submitAndWait` loops and all `Thread.sleep` calls with these reactive handles.

- [ ] **Step 6: Run unit and real persistence-race tests**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=AgentScopeToolAdapterTests,ImageGenerationConsumerTests,VideoGenerationConsumerTests" test`
Expected: PASS; the executor runs on `agent-tool-blocking-*`, timeout/cancel/fence loss cancel and close the handle once, and no success is returned afterward.

Run: `.\mvnw.cmd -Pagentscope-integration -Dfailsafe.failIfNoSpecifiedTests=true -Dit.test=AgentToolCancellationIT verify`
Expected: PASS against MySQL/Redis containers; cancel and remote completion races leave exactly one terminal task status, and a cancelled image/video task has no successful artifact URL.

- [ ] **Step 7: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeToolAdapter.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/ToolExecutionContext.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/ToolExecutionHandle.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/ToolCancellationReason.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/ToolExecutionRequest.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/InProcessToolExecutionHandle.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/CancellableToolExecutor.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/CancellableToolExecutionPort.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/DefaultCancellableToolExecutionPort.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/tool/GenerateImageToolExecutor.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/tool/GenerateVideoToolExecutor.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/generation/consumer/ImageGenerationConsumer.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/generation/consumer/VideoGenerationConsumer.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/generation/ImageGenerationService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/generation/VideoGenerationService.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeToolAdapterTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/generation/consumer/ImageGenerationConsumerTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/generation/consumer/VideoGenerationConsumerTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentToolCancellationIT.java
git commit -m "feat: enforce tool deadline cancellation and fencing"
```

### Task 10: Platform Sub-Agent Identity and Cancellation

**Files:**
- Replace: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeSubAgentToolAdapter.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformSubAgentRunPort.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformSubAgentCommand.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformSubAgentRun.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/PlatformSubAgentRunService.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/context/ToolExecutionContext.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeSubAgentToolAdapterTests.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeSubAgentWiringTests.java`

**Interfaces:** Consumes the exact Durable Runtime-owned `PlatformSubAgentCommand(parentRunId,parentOwnerInstanceId,parentOwnerEpoch,parentToolCallId,agentName,kernelSpec,messages,projectContext,deadline)`, `PlatformSubAgentRun(childRunId,parentRunId,parentToolCallId,agentName,status)`, `PlatformSubAgentRunPort.start(...)`, and `cancelChildren(parentRunId)`. The adapter resolves the typed `ToolExecutionContext` from `ToolCallParam#getRuntimeContext()` and copies the parent owner tuple from the same RuntimeContext's `AgentRunContext`; the command deliberately exposes no `sessionId`. `PlatformSubAgentRun` exposes the platform `childRunId` and durable `AgentRunStatus`; AgentScope native `taskId` is never serialized into tool output or persisted as a child run ID.

- [ ] **Step 1: Write failing parent/child identity and cancellation tests**

```java
@Test void childForwardsParentFenceAndNeverExposesNativeTaskId() {
    ToolExecutionContext toolContext = mock(ToolExecutionContext.class);
    ToolCallParam param = toolParamWithRuntime(
        new AgentRunContext("parent-1", "node-a", 7L, deadline), toolContext);
    when(port.start(any())).thenReturn(Mono.just(new PlatformSubAgentRun(
        "child-1", "parent-1", "tool-9", "storyboard_frame_gen", RUNNING)));

    StepVerifier.create(adapter.callAsync(param)).assertNext(result -> {
        assertThat(text(result)).contains("childRunId", "child-1", "status", "RUNNING");
        assertThat(text(result)).doesNotContain("taskId");
    }).verifyComplete();

    verify(port).start(argThat(c -> c.parentRunId().equals("parent-1")
        && c.parentOwnerInstanceId().equals("node-a")
        && c.parentOwnerEpoch() == 7L
        && c.parentToolCallId().equals("tool-9")));
}
@Test void springUsesTheDurablePortImplementation() {
    assertThat(context.getBean(PlatformSubAgentRunPort.class).getClass().getName())
        .isEqualTo("com.stonewu.fusion.service.ai.run.PlatformSubAgentRunService");
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=AgentScopeSubAgentToolAdapterTests,AgentScopeSubAgentWiringTests" test`
Expected: FAIL because the old adapter creates a V1 `ReActAgent` directly or a shadow/local Port is used instead of the Durable Runtime bean.

- [ ] **Step 3: Delegate to the durable child-run port**

```java
public Mono<ToolResultBlock> callAsync(ToolCallParam param) {
    ToolRuntime runtime = requireRuntime(param.getRuntimeContext());
    String parentToolCallId = param.getToolUseBlock().getId();
    String childSession = sessionIds.forAgent(runtime.conversationId(), definition.getRefAgentType());
    PlatformSubAgentCommand command = new PlatformSubAgentCommand(
        runtime.run().runId(), runtime.run().ownerInstanceId(), runtime.run().ownerEpoch(),
        parentToolCallId, definition.getRefAgentType(),
        definition.toKernelSpec(childSession), List.of(inputMessage(param)),
        runtime.project(), runtime.run().deadline());
    return port.start(command).map(child -> toolResult(param, Map.of(
        "childRunId", child.childRunId(),
        "agentName", child.agentName(),
        "status", child.status().name())));
}
```

- [ ] **Step 4: Run sub-agent tests**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=AgentScopeSubAgentToolAdapterTests,AgentScopeSubAgentWiringTests" test`
Expected: PASS for `asset_image_gen`, `storyboard_frame_gen`, `storyboard_video_gen`, Durable Port wiring, the exact parent owner tuple copied from the `ToolCallParam` RuntimeContext while the typed `ToolExecutionContext` is present, and output containing the durable `childRunId/status` but no native `taskId`. The unit test does not inspect command session identity; independent child-session derivation is proved by `PlatformSubAgentRunServiceIT`.

Run: `.\mvnw.cmd -Pagentscope-integration -Dfailsafe.failIfNoSpecifiedTests=true -Dit.test=PlatformSubAgentRunServiceIT verify`
Expected: PASS; the Durable Runtime-owned test proves parent/child persistence and `CancellationCoordinator.cancel(parentRunId,userId)` invokes `cancelChildren` across nodes.

- [ ] **Step 5: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeSubAgentToolAdapter.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeSubAgentToolAdapterTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeSubAgentWiringTests.java
git commit -m "feat: run platform sub agents with durable identity"
```

### Task 11: Recoverable Confirmation and External Waiting

**Files:**
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/AgentKernelSnapshot.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/AgentKernelSnapshotBuilder.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/AgentKernelSnapshotResolver.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/StartRootAgentRequest.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/waiting/AgentResumeMessageFactory.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/waiting/UserConfirmationCoordinator.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/waiting/ExternalExecutionCoordinator.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/security/InternalAgentExecutorAuthorizer.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AgentConfirmReqVO.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AgentConfirmDecisionVO.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AgentExternalResultReqVO.java`
- Modify: `ai-fusion-video/src/main/resources/application.yaml`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/AiPipelineController.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantService.java`
- Modify: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRunQueryService.java`
- Create: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/AuthorizedInternalAgentRun.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentExecutionRuntimeContextRequests.java`
- Consume: `ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/RunExecutionSupervisor.java`
- Consume: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/kernel/AgentKernelSnapshotContractTests.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/waiting/WaitingCoordinatorTests.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/controller/ai/AiPipelineWaitingControllerTests.java`
- Test: `ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentWaitingHttpResumeIT.java`

**Interfaces:** Consumes Durable Runtime's `AgentKernelSnapshotBuilder.build(AgentKernelSpec)`, `AgentKernelSnapshotResolver.resolve(AgentKernelSnapshot)`, frozen `AgentWaitingStatePort`, `AgentRunQueryService.requireInternalExecutorRun(runId,executorId)`, `Mono<AgentScopeRuntimeContextRequest> AgentExecutionRuntimeContextRequests.forResume(ResumedAgentRun)`, StateStore guard, and `RunExecutionSupervisor.resume(ResumeAgentExecutionCommand)`. `POST /api/ai/pipeline/confirm` authenticates the current user; `POST /api/ai/pipeline/external-result` authenticates an internal executor. Both rebuild only server-persisted identities and the original Kernel snapshot.

- [ ] **Step 1: Write failing save-order, authorization, and cross-node tests**

```java
@Test void confirmationIsNotActionableUntilNaturalResultAndStateSave() {
    StepVerifier.create(coordinator.recordCandidate("run-1", candidate)).verifyComplete();
    verify(eventPublisher, never()).publishActionable(any());
    StepVerifier.create(coordinator.afterSavedResult(runtimeContext, checkpoint)).expectNext(true).verifyComplete();
    verify(eventPublisher).publishActionable(any());
}
@Test void externalResultRebuildsServerPersistedToolIdentity() {
    StepVerifier.create(external.resume(request, "executor-a")).verifyComplete();
    verify(supervisor).resume(argThat(command -> command.ownerEpoch() == 8
        && toolId(command.messages()).equals("tool-4")));
}
@Test void snapshotRejectsTamperingAndUnavailableManifest() {
    AgentKernelSnapshot built = snapshots.build(spec);
    StepVerifier.create(resolver.resolve(new AgentKernelSnapshot(
            "0".repeat(64), built.snapshotJson())))
        .expectError(RunConfigUnavailableException.class).verify();
    StepVerifier.create(resolver.resolve(new AgentKernelSnapshot(
            built.fingerprint(), jsonWithManifest("removed-tools-v0"))))
        .expectError(RunConfigUnavailableException.class).verify();
}
@Test void confirmControllerNeverAcceptsClientToolInput() throws Exception {
    mockMvc.perform(post("/api/ai/pipeline/confirm")
            .contentType(APPLICATION_JSON)
            .content("{\"runId\":\"run-1\",\"replyId\":\"reply-4\",\"decisions\":[{\"toolCallId\":\"tool-4\",\"confirmed\":true}],\"input\":{\"unsafe\":true}}"))
        .andExpect(status().isBadRequest());
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=AgentKernelSnapshotContractTests,WaitingCoordinatorTests,AiPipelineWaitingControllerTests" test`
Expected: FAIL because the coordinators, strict request VOs, authorized endpoints, and resume-to-supervisor mapping do not exist; Durable Runtime snapshot contracts must already pass.

- [ ] **Step 3: Feed the Durable Runtime-owned immutable snapshot into root execution**

```java
public record StartRootAgentRequest(
    String runId, String conversationId, long userId, Long projectId,
    String agentType, String sessionId, String userContent, String referencesJson,
    List<Msg> messages, Instant deadline) {}

public Mono<StartedAgentRun> startRoot(StartRootAgentRequest request) {
    AgentKernelSpec spec = kernelSpecs.forRequest(request);
    AgentKernelSnapshot snapshot = snapshotBuilder.build(spec);
    StartAgentRunCommand start = new StartAgentRunCommand(
        request.runId(), request.conversationId(), request.userId(), request.projectId(),
        request.agentType(), null, null, null, request.sessionId(), snapshot,
        instanceId, ownerLease, request.userContent(), request.referencesJson());
    return runCoordinator.start(start)
        .flatMap(started -> {
            AgentScopeRuntimeContextRequest runtime = runtimeContextRequests.forRoot(
                request, started, request.deadline());
            StartAgentExecutionCommand execution = new StartAgentExecutionCommand(
                started, request.messages(), snapshot, spec, runtime);
            return supervisor.start(execution).thenReturn(started);
        });
}
```

The snapshot is built exactly once before the start transaction and the same value is passed to `StartAgentRunCommand` and `StartAgentExecutionCommand`. Resume never recomputes it; `DefaultRunExecutionSupervisor.resume` calls the Durable Runtime resolver and maps incompatibility to `FAILED/RUN_CONFIG_UNAVAILABLE`.

- [ ] **Step 4: Implement candidate/save/CAS sequencing without duplicate publication**

```java
public Mono<Boolean> afterSavedResult(RuntimeContext runtime, WaitingCheckpoint checkpoint) {
    AgentRunContext run = Objects.requireNonNull(runtime.get(AgentRunContext.class));
    return Mono.fromRunnable(() -> stateGuard.throwIfFailed(
            new StateStoreSlot(runtime.getUserId(), checkpoint.sessionId())))
        .then(waitingPort.enterWaitingConfirmation(run.runId(), run.ownerEpoch(), checkpoint))
        .flatMap(won -> won ? Mono.just(true) : Mono.just(false));
}

public Mono<Boolean> afterSavedExternalResult(
        RuntimeContext runtime, WaitingCheckpoint checkpoint, PendingExternalExecution pending) {
    AgentRunContext run = Objects.requireNonNull(runtime.get(AgentRunContext.class));
    return Mono.fromRunnable(() -> stateGuard.throwIfFailed(
            new StateStoreSlot(runtime.getUserId(), checkpoint.sessionId())))
        .then(waitingPort.enterWaitingExternal(run.runId(), run.ownerEpoch(), checkpoint, pending));
}
```

`enterWaitingConfirmation/enterWaitingExternal` already insert the actionable `PLATFORM_*` journal event and outbox row in their CAS transaction; this task must not publish a second event. The Harness lease is released only after these methods return `true`.

- [ ] **Step 5: Resume confirmation and external results through the frozen supervisor command**

```java
public Mono<Void> resume(ConfirmRequest request, long currentUserId) {
    ResumeConfirmationCommand command = commands.confirmation(request, currentUserId);
    return waitingPort.resumeConfirmation(command)
        .flatMap(resumed -> resume(resumed, resumeMessages.confirmation(request.decisions())));
}

private Mono<Void> resume(ResumedAgentRun resumed, List<Msg> messages) {
    AgentKernelSnapshot snapshot = new AgentKernelSnapshot(
        resumed.kernelFingerprint(), resumed.agentDefinitionSnapshotJson());
    return runtimeContextRequests.forResume(resumed)
        .flatMap(runtime -> supervisor.resume(new ResumeAgentExecutionCommand(
            resumed, messages, snapshot, runtime)));
}

public Mono<Void> resume(AgentExternalResultReqVO request, String executorId) {
    return runQueries.requireInternalExecutorRun(request.runId(), executorId)
        .flatMap(run -> waitingPort.resumeExternal(commands.external(
            request, run.userId(), executorId)))
        .flatMap(resumed -> resume(resumed, resumeMessages.external(
            request.toolCallId(), request.resultPayloadJson())));
}

public Mono<AuthorizedInternalAgentRun> requireInternalExecutorRun(
        String runId, String executorId) {
    return Mono.fromCallable(() -> runRepository.requireWaitingExternal(runId))
        .subscribeOn(schedulers.journal())
        .map(run -> new AuthorizedInternalAgentRun(
            run.getRunId(), run.getUserId(), executorId, run.getWaitingToolCallId(),
            run.getWaitExpiresAt().toInstant(ZoneOffset.UTC)));
}

public record AuthorizedInternalAgentRun(
    String runId, long userId, String executorId,
    String toolCallId, Instant expiresAt) {}
```

`AgentResumeMessageFactory` loads the persisted pending event and constructs `Msg.METADATA_CONFIRM_RESULTS` or the matching `ToolResultBlock`; it ignores client tool name, input, schema, reply payload identity, and all unexpected JSON properties.

- [ ] **Step 6: Expose authorized HTTP endpoints with strict VOs**

```java
public record AgentConfirmDecisionVO(String toolCallId, boolean confirmed) {}
public record AgentConfirmReqVO(
    String runId, String replyId, List<AgentConfirmDecisionVO> decisions) {}
public record AgentExternalResultReqVO(
    String runId, String toolCallId, String resultPayloadJson) {}

@Component
public final class InternalAgentExecutorAuthorizer {
    private final byte[] expectedToken;

    public InternalAgentExecutorAuthorizer(
            @Value("${fusion.agentscope.v2.internal-executor-token}") String expectedToken) {
        if (expectedToken == null || expectedToken.isBlank()) {
            throw new IllegalStateException("AFV_INTERNAL_EXECUTOR_TOKEN is required");
        }
        this.expectedToken = expectedToken.getBytes(StandardCharsets.UTF_8);
    }

    public String requireExecutorId(HttpServletRequest request) {
        String executorId = request.getHeader("X-AFV-Executor-Id");
        String supplied = request.getHeader("X-AFV-Executor-Token");
        if (executorId == null || executorId.isBlank() || supplied == null
                || !MessageDigest.isEqual(expectedToken,
                    supplied.getBytes(StandardCharsets.UTF_8))) {
            throw new BusinessException(403, "INTERNAL_EXECUTOR_UNAUTHORIZED");
        }
        return executorId;
    }
}

// application.yaml: fusion.agentscope.v2.internal-executor-token=${AFV_INTERNAL_EXECUTOR_TOKEN}

@PostMapping("/confirm")
public Mono<CommonResult<Void>> confirm(@Valid @RequestBody AgentConfirmReqVO request) {
    long currentUserId = requireCurrentUserId();
    return confirmations.resume(ConfirmRequest.from(request), currentUserId)
        .thenReturn(CommonResult.success(null));
}

@PostMapping("/external-result")
public Mono<CommonResult<Void>> externalResult(
        @Valid @RequestBody AgentExternalResultReqVO request,
        HttpServletRequest servletRequest) {
    String executorId = internalExecutorAuthorizer.requireExecutorId(servletRequest);
    return externalExecutions.resume(request, executorId).thenReturn(CommonResult.success(null));
}
```

Configure Jackson to fail on unknown properties for these request records. Confirm must join `conversation.user_id`, require `WAITING_CONFIRMATION`, matching replyId, unexpired wait, and an exact decision/toolCallId set. External result requires constant-time verification of `X-AFV-Executor-Token`, a nonblank `X-AFV-Executor-Id`, `WAITING_EXTERNAL`, matching toolCallId, and an unexpired wait. Neither endpoint logs the token or result payload.

- [ ] **Step 7: Run unit and cross-node HTTP recovery tests**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=AgentKernelSnapshotContractTests,WaitingCoordinatorTests,AiPipelineWaitingControllerTests,RunExecutionSupervisorTests" test`
Expected: PASS for snapshot tamper/redaction, crash-before-save, approval/rejection, expiry, duplicate response, exact decision set, internal-only external result, `TOOL_SUSPENDED`, and `RUN_CONFIG_UNAVAILABLE`.

Run: `.\mvnw.cmd -Pagentscope-integration -Dfailsafe.failIfNoSpecifiedTests=true -Dit.test=AgentWaitingHttpResumeIT verify`
Expected: PASS with two real application instances: node A reaches WAITING and releases its lease; HTTP confirm/external-result reaches node B; B performs user/internal authorization, CAS, snapshot rebuild, and `RunExecutionSupervisor.resume(ResumeAgentExecutionCommand)`; stale A is fenced; replay exposes one terminal event.

- [ ] **Step 8: Commit**

```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/StartRootAgentRequest.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/waiting ai-fusion-video/src/main/java/com/stonewu/fusion/security/InternalAgentExecutorAuthorizer.java ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AgentConfirmReqVO.java ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AgentConfirmDecisionVO.java ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/vo/AgentExternalResultReqVO.java ai-fusion-video/src/main/java/com/stonewu/fusion/controller/ai/AiPipelineController.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRunQueryService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/AuthorizedInternalAgentRun.java ai-fusion-video/src/main/resources/application.yaml ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/agentscope/waiting/WaitingCoordinatorTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/controller/ai/AiPipelineWaitingControllerTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentWaitingHttpResumeIT.java
git commit -m "feat: persist and resume AgentScope waiting states"
```

### Task 12: Provider and Ark Smoke Profiles plus Final Verification

**Files:**
- Modify: `ai-fusion-video/pom.xml`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/smoke/AgentScopeProviderSmokeIT.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/generation/smoke/ArkMediaSmokeIT.java`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/smoke/SmokeEnvironment.java`

**Interfaces:** `agentscope-provider-smoke` runs five real minimal streams; `ark-smoke` runs one real image request and one real video task creation. Required environment is validated before network access.

- [ ] **Step 1: Write the credential validator and profile-selection tests**

```java
static String require(String name) {
    String value = System.getenv(name);
    if (value == null || value.isBlank()) throw new IllegalStateException("Missing required smoke credential: " + name);
    return value;
}

@ParameterizedTest
@ValueSource(strings = {"openai", "anthropic", "gemini", "dashscope", "ollama"})
void realProviderCompletesOneMinimalStream(String provider) {
    try (OwnedChatModel model = factory.create(smokeKernelSpec(provider))) {
        ChatResponse response = withRuntime(model.model().stream(List.of(UserMessage.builder().textContent("Reply OK").build()), List.of(), null))
            .blockLast(Duration.ofMinutes(2));
        assertNotNull(response);
    }
}
```

- [ ] **Step 2: Run profiles without credentials and confirm fast failure**

Run: `.\mvnw.cmd -Pagentscope-provider-smoke -Dfailsafe.failIfNoSpecifiedTests=true verify; .\mvnw.cmd -Park-smoke -Dfailsafe.failIfNoSpecifiedTests=true verify`
Expected: FAIL with `Missing required smoke credential`, not SKIPPED.

- [ ] **Step 3: Add isolated Failsafe profiles**

```xml
<profile><id>agentscope-provider-smoke</id><build><plugins><plugin>
  <groupId>org.apache.maven.plugins</groupId><artifactId>maven-failsafe-plugin</artifactId>
  <executions><execution><goals><goal>integration-test</goal><goal>verify</goal></goals>
  <configuration><failIfNoTests>true</failIfNoTests><includes><include>**/AgentScopeProviderSmokeIT.java</include></includes></configuration></execution></executions>
</plugin></plugins></build></profile>
<profile><id>ark-smoke</id><build><plugins><plugin>
  <groupId>org.apache.maven.plugins</groupId><artifactId>maven-failsafe-plugin</artifactId>
  <executions><execution><goals><goal>integration-test</goal><goal>verify</goal></goals>
  <configuration><failIfNoTests>true</failIfNoTests><includes><include>**/ArkMediaSmokeIT.java</include></includes></configuration></execution></executions>
</plugin></plugins></build></profile>
```

- [ ] **Step 4: Run all non-secret verification, then credentialed smoke**

Run: `.\mvnw.cmd -Dsurefire.failIfNoSpecifiedTests=true "-Dtest=DefaultModelCallUsageLedgerTests,ModelCallUsageSettlementWorkerTests,ManagedAgentScopeChatModelTests,AgentScopeGaProviderFactoryTests,OpenAiResponsesAgentScopeModelTests,ProviderCapabilityEvidenceTests,SafeAgentMediaResolverTests,AgentScopeMessageMapperTests,PlatformToolRegistryTests,AgentScopeToolHarnessIntegrationTests,AgentScopeToolAdapterTests,ImageGenerationConsumerTests,VideoGenerationConsumerTests,AgentScopeSubAgentToolAdapterTests,AgentScopeSubAgentWiringTests,WaitingCoordinatorTests,AiPipelineWaitingControllerTests" test`
Expected: PASS with every named test class executed; a missing class fails instead of being silently ignored.

Run: `.\mvnw.cmd -Pagentscope-integration -Dfailsafe.failIfNoSpecifiedTests=true "-Dit.test=AgentToolCancellationIT,PlatformSubAgentRunServiceIT,AgentWaitingHttpResumeIT" verify`
Expected: PASS with real MySQL/Redis containers and two-node WAITING/sub-agent behavior.

Run: `.\mvnw.cmd dependency:tree -Dincludes=io.agentscope; .\mvnw.cmd test; .\mvnw.cmd package`
Expected: PASS. Then run both smoke profiles with documented `AFV_SMOKE_*` variables; expected PASS with one real stream per Provider and real Ark image/video responses. If credentials are unavailable, record both profiles as UNVERIFIED.

- [ ] **Step 5: Commit**

```powershell
git add ai-fusion-video/pom.xml ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/provider/smoke ai-fusion-video/src/test/java/com/stonewu/fusion/service/generation/smoke
git commit -m "test: add AgentScope provider and Ark smoke profiles"
```

## Final Review Gate

- [ ] `rg -n "\.block\(|\.toIterable\(|Thread\.sleep\(|ThreadLocal|Schedulers\.boundedElastic\(" ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/provider ai-fusion-video/src/main/java/com/stonewu/fusion/service/generation/consumer` returns no production violations.
- [ ] `rg -n "apiKey|proxyPassword|Authorization|base64"` review confirms values are not logged or persisted.
- [ ] `dependency:tree` contains only AgentScope `2.0.0` GA modules.
- [ ] A ReAct run with two model calls creates two usage rows and settles both with distinct idempotency keys.
- [ ] Complete/error/cancel close all model streams, clients, tools, and child-run handles exactly once.
- [ ] WAITING transitions occur only after natural result plus successful AgentState save, and cross-node resume uses the persisted snapshot and session.
- [ ] `AgentScopeModelFactory` is the only `AgentKernelModelFactory` bean; Vertex implements the same `ChatModelBase` contract as the five official extensions.
- [ ] The production Harness receives only manifest/whitelist-approved tools, and image/video cancellation wins or suppresses every late success backfill.
- [ ] Platform sub-agent adapter tests prove typed `ToolExecutionContext` resolution plus parent owner-tuple forwarding from `AgentRunContext`, output uses durable `childRunId/status`, and native `taskId` is never exposed.
- [ ] Confirm and external-result HTTP tests prove user/internal authorization, strict request identity, expiry/duplicate CAS, and `ResumeAgentExecutionCommand` delivery.
- [ ] No Migration, entity, mapper, or Durable Runtime repository file is changed by this plan.
