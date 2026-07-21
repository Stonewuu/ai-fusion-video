# AgentScope V2 GA Durable Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MySQL-backed AgentScope V2 run journal, transactional terminal/outbox/projection pipeline, fenced multi-node execution, recoverable WAITING states, and cursor-correct backend SSE contract required by P-1.

**Architecture:** MySQL is authoritative for run state, per-run event sequence, model-call usage, WAITING checkpoints, message projection, terminal CAS, and replay. Redis carries only wake-up/cancel signals; live delivery always re-reads committed MySQL rows. `RunExecutionSupervisor` owns execution independently from HTTP observers and bounds ingress by event count and bytes.

**Tech Stack:** Java 21, Spring Boot 3.5.14, Reactor, MyBatis-Plus 3.5.12, Flyway, MySQL 8.0.16+, Spring Data Redis, JUnit 5, Testcontainers, Maven Surefire/Failsafe.

## Global Constraints

- Add only `V1.0.6.1.5__agent_run_and_event.sql`; never edit executed migrations.
- MySQL must be `>=8.0.16`; use a maintenance write window for migration and V1/V2 writer switches.
- Keep `next_message_order BIGINT NOT NULL DEFAULT 1` through the rollback window.
- MySQL event rows are replay truth; Redis IDs/payloads are never product cursors or ordered facts.
- Per-run sequence begins at 1; raw-only events consume sequence and may create legal SSE gaps.
- Never use `.block()`, `.toIterable()`, `Thread.sleep()`, or `ThreadLocal` in production paths.
- MyBatis calls run on bounded `agent-journal`; HTTP disconnect never cancels business execution.
- Exactly one of `COMPLETED/FAILED/CANCELLED` and one matching terminal event may win.
- `active_conversation_id` serializes root runs only (`parent_run_id IS NULL`); platform child runs share the parent conversation, are indexed by `parent_run_id`, and may execute beneath that one active root.
- Every run persists a non-null `deadline_at`; child deadline is never later than its locked parent deadline.
- `run/cancel/status/running/reconnect/confirm` authorize current user through run and conversation ownership; external-result requires the internal executor identity.
- SSE data frames use `id: {runId}:{sequence}`; `Last-Event-ID` and `afterSequence` must agree.
- Frontend changes are excluded; this plan freezes and tests only the backend SSE contract.
- P-1 schedules no event-history deletion. A future retention worker may select only terminal runs where `projected_through_sequence >= terminal_sequence` and `projection_completed_at IS NOT NULL`; conversation logical deletion, Redis AgentState session cleanup, audit retention, and a user erasure request remain separate operations.
- The dependency/kernel plan owns `service/ai/run/AgentRuntimeShutdownPort.java`. Durable Runtime owns `service/ai/run/model/StartAgentExecutionCommand.java`, `ResumeAgentExecutionCommand.java`, `ExecutionStopReason.java`, `service/ai/run/RunExecutionSupervisor.java`, and `CancellationCoordinator.java`; `RunExecutionSupervisor extends AgentRuntimeShutdownPort`, and no plan may create shadows under `service/ai/agentscope/runtime`.
- Durable Runtime owns `PlatformSubAgentRunPort`, its command/result records, and the production `PlatformSubAgentRunService`; the model/tool slice only adapts ToolBase calls to this port.
- Durable Runtime owns `ModelUsageSettlementPort`, the production audit-ledger adapter, and its Spring binding; the model/tool slice owns only the settlement worker that calls this port.
- Unless a step names another directory, every Maven command runs from `D:\develop\my\ai-fusion-video\ai-fusion-video`; every focused Failsafe command names its IT with `-Dit.test` and the profile fails when that named test does not exist.

Path roots used in every abbreviated `Files` entry below are exact: `service/...`, `repository/...`, `entity/...`, `mapper/...`, `model/...`, and `controller/...` resolve under `ai-fusion-video/src/main/java/com/stonewu/fusion/`; `test/...` resolves under `ai-fusion-video/src/test/java/com/stonewu/fusion/`; `integration/...` resolves under `ai-fusion-video/src/test/java/com/stonewu/fusion/integration/`. No other base directory is implied.

## Frozen Interfaces

```java
public record PendingConfirmation(String replyId, Set<String> decisionIds,
    String pendingToolCallsJson, Instant expiresAt) {}
public record PendingExternalExecution(String toolCallId, String toolName,
    String suspendedPayloadJson, Instant expiresAt) {}
public record WaitingCheckpoint(String sessionId, String kernelFingerprint,
    String agentDefinitionSnapshotJson, long pausedThroughSequence) {}
public record ResumeConfirmationCommand(String runId, long currentUserId,
    String replyId, Set<String> decisionIds, String newOwnerInstanceId,
    Duration ownerLease) {}
public record ResumeExternalCommand(String runId, long currentUserId,
    String internalExecutorId, String toolCallId, String resultPayloadJson,
    String newOwnerInstanceId, Duration ownerLease) {}
public record ResumedAgentRun(String runId, String conversationId, String sessionId,
    String kernelFingerprint, String agentDefinitionSnapshotJson,
    long pausedThroughSequence, String newOwnerInstanceId,
    long newOwnerEpoch, Instant leaseUntil, Instant deadline) {}

public interface AgentWaitingStatePort {
    Mono<Void> recordConfirmationCandidate(String runId, PendingConfirmation candidate);
    Mono<Boolean> enterWaitingConfirmation(String runId, long expectedOwnerEpoch,
        WaitingCheckpoint checkpoint);
    Mono<Boolean> enterWaitingExternal(String runId, long expectedOwnerEpoch,
        WaitingCheckpoint checkpoint, PendingExternalExecution pending);
    Mono<PendingConfirmation> getPendingConfirmationAuthorized(String runId,
        long currentUserId, String replyId);
    Mono<PendingExternalExecution> getPendingExternalAuthorized(String runId,
        long currentUserId, String toolCallId);
    Mono<ResumedAgentRun> resumeConfirmation(ResumeConfirmationCommand command);
    Mono<ResumedAgentRun> resumeExternal(ResumeExternalCommand command);
}

public interface AgentModelCallUsageRepository {
    void startCall(String runId, String modelCallId, String provider, String modelCode);
    boolean completeCall(String runId, String modelCallId, NormalizedModelUsage usage);
    boolean failCall(String runId, String modelCallId, AgentModelCallStatus status);
    List<AgentModelCallUsage> claimSettlementBatch(String claimOwner, Duration claimLease, int limit);
    boolean markSettled(long usageId, String claimOwner, String downstreamSettlementId);
    boolean releaseSettlementForRetry(long usageId, String claimOwner,
        Instant nextAttemptAt, String sanitizedError);
    boolean markRunUsageSettledIfAllCallsSettled(String runId);
}

public record AgentKernelSnapshot(String fingerprint, String snapshotJson) {
    public AgentKernelSnapshot {
        if (fingerprint == null || !fingerprint.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException("kernel fingerprint must be lowercase SHA-256");
        }
        if (snapshotJson == null || snapshotJson.isBlank()) {
            throw new IllegalArgumentException("kernel snapshot JSON is required");
        }
    }
}

public interface AgentKernelSnapshotBuilder {
    AgentKernelSnapshot build(AgentKernelSpec spec);
}

public interface AgentKernelSnapshotResolver {
    Mono<AgentKernelSpec> resolve(AgentKernelSnapshot snapshot);
}

public record StartAgentExecutionCommand(
    StartedAgentRun run,
    List<Msg> messages,
    AgentKernelSnapshot kernelSnapshot,
    AgentKernelSpec kernelSpec,
    AgentScopeRuntimeContextRequest runtimeContextRequest) {}

public record ResumeAgentExecutionCommand(
    ResumedAgentRun run,
    List<Msg> messages,
    AgentKernelSnapshot kernelSnapshot,
    AgentScopeRuntimeContextRequest runtimeContextRequest) {}

public enum ExecutionStopReason { CANCEL_REQUESTED, OWNER_FENCED, DEADLINE, SHUTDOWN }

public interface RunExecutionSupervisor extends AgentRuntimeShutdownPort {
    Mono<Void> start(StartAgentExecutionCommand command);
    Mono<Void> resume(ResumeAgentExecutionCommand command);
    Mono<Boolean> interruptOwned(String runId, String ownerInstanceId,
        long ownerEpoch, ExecutionStopReason reason);
}

public interface RunShutdownCancellationPort {
    Mono<Void> request(String runId);
}

public interface AgentEventJournal {
    Mono<Optional<CommittedAgentEvent>> appendOwned(String runId,
        String ownerInstanceId, long ownerEpoch, AgentEventEnvelope event);
}

public enum SystemTerminalActor { CANCELLATION_COORDINATOR, OWNER_RECONCILER }

public interface RunTerminalCoordinator {
    Mono<Optional<CommittedAgentEvent>> terminateOwned(RunTerminalRequest request,
        String ownerInstanceId, long ownerEpoch);
    Mono<Optional<CommittedAgentEvent>> terminateSystem(RunTerminalRequest request,
        SystemTerminalActor actor);
}

public record PlatformSubAgentCommand(
    String parentRunId,
    String parentToolCallId,
    String parentOwnerInstanceId,
    long parentOwnerEpoch,
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

public interface ModelUsageSettlementPort {
    Mono<String> settle(String idempotencyKey, NormalizedModelUsage usage);
}
```

---

### Task 1: Add Flyway schema and Failsafe foundation

**Files:**
- Modify: `ai-fusion-video/pom.xml`
- Create: `ai-fusion-video/src/main/resources/db/migration/V1.0.6.1.5__agent_run_and_event.sql`
- Create: `ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentPersistenceMigrationIT.java`

**Interfaces:** Produces the `agentscope-integration` profile and the frozen run/event/usage/message schema consumed by all later tasks. Every run has `deadline_at DATETIME(3) NOT NULL`; `idx_agent_run_status_deadline(status,deadline_at,id)` drives deadline maintenance. Child runs persist `parent_run_id`, `parent_tool_call_id`, and `agent_name`; `uk_agent_run_parent_tool(parent_run_id,parent_tool_call_id)` makes one child admission idempotent, while `idx_agent_run_parent_status(parent_run_id,status,id)` is the parent-cancellation scan path. The generated active conversation key applies only to root rows so a parent and its child runs can coexist without weakening the one-active-root invariant.

- [ ] Write the failing Testcontainers migration test:
```java
@Testcontainers(disabledWithoutDocker = false)
class AgentPersistenceMigrationIT {
  @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6");
  @Test void createsRunEventUsageAndOrderingConstraints() {
    migrate(MYSQL);
    assertThat(tableCount("afv_agent_run","afv_agent_event","afv_agent_model_call_usage")).isEqualTo(3);
    assertThat(indexColumns("afv_agent_message","uk_agent_message_conv_order"))
        .containsExactly("conversation_id","message_order");
    assertThat(columnDefault("afv_agent_conversation","next_message_order")).isEqualTo("1");
    assertThat(columns("afv_agent_run"))
        .contains("parent_run_id","parent_tool_call_id","agent_name",
            "deadline_at","projected_through_sequence","projection_completed_at");
    assertThat(isNullable("afv_agent_run","deadline_at")).isFalse();
    assertThat(indexColumns("afv_agent_run","uk_agent_run_parent_tool"))
        .containsExactly("parent_run_id","parent_tool_call_id");
    assertThat(indexColumns("afv_agent_run","idx_agent_run_parent_status"))
        .containsExactly("parent_run_id","status","id");
    assertThat(indexColumns("afv_agent_run","idx_agent_run_status_deadline"))
        .containsExactly("status","deadline_at","id");
    assertThat(columns("afv_agent_event"))
        .contains("publish_required","publish_status","publish_claim_owner",
            "publish_claim_until","next_publish_attempt_at");
    insertRunningRoot("conv-1","root-1");
    assertThatCode(() -> insertRunningChild("conv-1","root-1","tool-1","child-a"))
        .doesNotThrowAnyException();
    assertThatThrownBy(() -> insertRunningChild("conv-1","root-1","tool-1","child-b"))
        .isInstanceOf(DuplicateKeyException.class);
    assertThatThrownBy(() -> insertRunningRoot("conv-1","root-2"))
        .isInstanceOf(DuplicateKeyException.class);
  }
}
```
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentPersistenceMigrationIT" verify`; expect FAIL because the target schema is absent.
- [ ] Add the Failsafe/Testcontainers profile and executable migration:
```xml
<profile>
  <id>agentscope-integration</id>
  <dependencies>
    <dependency>
      <groupId>org.testcontainers</groupId><artifactId>junit-jupiter</artifactId>
      <version>1.21.3</version><scope>test</scope>
    </dependency>
    <dependency>
      <groupId>org.testcontainers</groupId><artifactId>mysql</artifactId>
      <version>1.21.3</version><scope>test</scope>
    </dependency>
    <dependency>
      <groupId>org.awaitility</groupId><artifactId>awaitility</artifactId>
      <version>4.3.0</version><scope>test</scope>
    </dependency>
  </dependencies>
  <build><plugins><plugin>
    <groupId>org.apache.maven.plugins</groupId><artifactId>maven-failsafe-plugin</artifactId>
    <version>3.5.4</version>
    <executions><execution><goals><goal>integration-test</goal><goal>verify</goal></goals></execution></executions>
    <configuration>
      <includes>
        <include>**/AgentPersistenceMigrationIT.java</include>
        <include>**/AgentMessageAllocatorIT.java</include>
        <include>**/AgentRunStartIT.java</include>
        <include>**/AgentJournalTerminalIT.java</include>
        <include>**/AgentModelCallUsageIT.java</include>
        <include>**/AgentOutboxMultiInstanceIT.java</include>
        <include>**/AgentProjectionRecoveryIT.java</include>
        <include>**/PlatformSubAgentRunServiceIT.java</include>
        <include>**/AgentWaitingStateIT.java</include>
        <include>**/AgentFencingCancellationIT.java</include>
        <include>**/AgentOwnedJournalTakeoverIT.java</include>
        <include>**/AgentReplayLiveIT.java</include>
        <include>**/AgentIntegrationProfileSentinelIT.java</include>
        <include>**/AgentDurableRuntimeMultiInstanceIT.java</include>
      </includes>
      <failIfNoTests>true</failIfNoTests>
      <failIfNoSpecifiedTests>true</failIfNoSpecifiedTests>
    </configuration>
  </plugin></plugins></build>
</profile>
```
```sql
CREATE TEMPORARY TABLE afv_order_rewrite(id BIGINT PRIMARY KEY,new_order BIGINT NOT NULL);
INSERT INTO afv_order_rewrite SELECT id,ROW_NUMBER() OVER(PARTITION BY conversation_id ORDER BY message_order,id) FROM afv_agent_message;
UPDATE afv_agent_message m JOIN afv_order_rewrite r ON r.id=m.id SET m.message_order=r.new_order;
DROP TEMPORARY TABLE afv_order_rewrite;
ALTER TABLE afv_agent_message MODIFY message_order BIGINT NOT NULL,
  ADD run_id VARCHAR(64) NULL, ADD projection_key VARCHAR(64) NULL,
  DROP INDEX idx_msg_conv_order,
  ADD UNIQUE KEY uk_agent_message_conv_order(conversation_id,message_order),
  ADD UNIQUE KEY uk_agent_message_projection_key(projection_key),
  ADD KEY idx_agent_message_conv_run_order(conversation_id,run_id,message_order);
ALTER TABLE afv_agent_conversation ADD next_message_order BIGINT NOT NULL DEFAULT 1;
UPDATE afv_agent_conversation c LEFT JOIN
 (SELECT conversation_id,COALESCE(MAX(message_order),0)+1 n,SUM(deleted=0) cnt FROM afv_agent_message GROUP BY conversation_id) m
 ON m.conversation_id=c.conversation_id SET c.next_message_order=COALESCE(m.n,1),c.message_count=COALESCE(m.cnt,0);
CREATE TABLE afv_agent_run(id BIGINT AUTO_INCREMENT PRIMARY KEY,run_id VARCHAR(64) NOT NULL,
 conversation_id VARCHAR(64) NOT NULL,user_id BIGINT NOT NULL,project_id BIGINT NULL,agent_type VARCHAR(64) NULL,
 parent_run_id VARCHAR(64) NULL,parent_tool_call_id VARCHAR(128) NULL,agent_name VARCHAR(128) NULL,
 kernel_fingerprint VARCHAR(64) NOT NULL,agent_definition_snapshot_json MEDIUMTEXT NOT NULL,
 agent_state_session_id VARCHAR(255) NOT NULL,status VARCHAR(32) NOT NULL,owner_instance_id VARCHAR(128) NULL,
 owner_epoch BIGINT NOT NULL DEFAULT 0,lease_until DATETIME(3) NULL,next_sequence BIGINT NOT NULL DEFAULT 1,
 terminal_sequence BIGINT NULL,terminal_output_type VARCHAR(32) NULL,cancel_requested_at DATETIME(3) NULL,
 cancel_broadcast_at DATETIME(3) NULL,cancel_acknowledged_at DATETIME(3) NULL,cancel_next_attempt_at DATETIME(3) NULL,
 waiting_reply_id VARCHAR(128) NULL,waiting_tool_call_id VARCHAR(128) NULL,waiting_tool_name VARCHAR(128) NULL,
 wait_expires_at DATETIME(3) NULL,paused_through_sequence BIGINT NULL,deadline_at DATETIME(3) NOT NULL,started_at DATETIME(3) NOT NULL,
 heartbeat_at DATETIME(3) NULL,finished_at DATETIME(3) NULL,error_code VARCHAR(64) NULL,error_message TEXT NULL,
 usage_settled TINYINT NOT NULL DEFAULT 0,usage_settled_at DATETIME(3) NULL,
 projected_through_sequence BIGINT NOT NULL DEFAULT 0,projection_completed_at DATETIME(3) NULL,
 active_conversation_id VARCHAR(64) GENERATED ALWAYS AS
  (CASE WHEN parent_run_id IS NULL AND status IN('RUNNING','WAITING_CONFIRMATION','WAITING_EXTERNAL','CANCEL_REQUESTED') THEN conversation_id END) STORED,
 create_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),update_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
 UNIQUE KEY uk_agent_run_id(run_id),UNIQUE KEY uk_agent_run_active(active_conversation_id),
 UNIQUE KEY uk_agent_run_parent_tool(parent_run_id,parent_tool_call_id),
 KEY idx_agent_run_conversation_status(conversation_id,status,id),
 KEY idx_agent_run_parent_status(parent_run_id,status,id),
 KEY idx_agent_run_user_status(user_id,status,update_time),
 KEY idx_agent_run_lease(status,lease_until),KEY idx_agent_run_status_deadline(status,deadline_at,id),
 KEY idx_agent_run_cancel(status,cancel_next_attempt_at),
 CONSTRAINT chk_agent_run_parent_identity CHECK((parent_run_id IS NULL AND parent_tool_call_id IS NULL)
   OR (parent_run_id IS NOT NULL AND parent_tool_call_id IS NOT NULL AND agent_name IS NOT NULL)),
 CONSTRAINT chk_agent_run_status CHECK(status IN('RUNNING','WAITING_CONFIRMATION','WAITING_EXTERNAL','CANCEL_REQUESTED','COMPLETED','FAILED','CANCELLED')));
CREATE TABLE afv_agent_event(id BIGINT AUTO_INCREMENT PRIMARY KEY,run_id VARCHAR(64) NOT NULL,sequence_no BIGINT NOT NULL,
 schema_version INT NOT NULL DEFAULT 1,raw_event_id VARCHAR(128) NULL,raw_event_type VARCHAR(64) NOT NULL,
 source VARCHAR(255) NULL,reply_id VARCHAR(128) NULL,block_id VARCHAR(128) NULL,tool_call_id VARCHAR(128) NULL,
 parent_tool_call_id VARCHAR(128) NULL,agent_name VARCHAR(128) NULL,output_type VARCHAR(32) NULL,payload_json MEDIUMTEXT NOT NULL,
 event_created_at DATETIME(3) NULL,redis_published_at DATETIME(3) NULL,publish_required TINYINT NOT NULL,
 publish_status VARCHAR(16) NOT NULL,publish_claim_owner VARCHAR(128) NULL,publish_claim_until DATETIME(3) NULL,
 next_publish_attempt_at DATETIME(3) NULL,last_publish_error VARCHAR(1024) NULL,publish_attempts INT NOT NULL DEFAULT 0,
 create_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),UNIQUE KEY uk_agent_event_sequence(run_id,sequence_no),
 KEY idx_agent_event_projection(run_id,output_type,sequence_no),KEY idx_agent_event_publish(publish_status,next_publish_attempt_at,id),
 KEY idx_agent_event_raw(run_id,raw_event_id),
 CONSTRAINT chk_agent_event_publish_status CHECK(publish_status IN('NOT_REQUIRED','PENDING','CLAIMED','PUBLISHED')),
 CONSTRAINT chk_agent_event_publish_required CHECK((publish_required=0 AND publish_status='NOT_REQUIRED') OR publish_required=1));
CREATE TABLE afv_agent_model_call_usage(id BIGINT AUTO_INCREMENT PRIMARY KEY,run_id VARCHAR(64) NOT NULL,
 model_call_id VARCHAR(64) NOT NULL,provider VARCHAR(64) NOT NULL,model_code VARCHAR(128) NOT NULL,status VARCHAR(24) NOT NULL,
 input_tokens BIGINT NULL,output_tokens BIGINT NULL,reasoning_tokens BIGINT NULL,cache_tokens BIGINT NULL,usage_json MEDIUMTEXT NULL,
 settlement_status VARCHAR(24) NOT NULL,settlement_attempts INT NOT NULL DEFAULT 0,next_settlement_attempt_at DATETIME(3) NULL,
 settlement_claim_owner VARCHAR(128) NULL,settlement_claim_until DATETIME(3) NULL,downstream_settlement_id VARCHAR(128) NULL,
 last_settlement_error VARCHAR(1024) NULL,started_at DATETIME(3) NOT NULL,finished_at DATETIME(3) NULL,
 create_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),update_time DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
 UNIQUE KEY uk_agent_usage_call(run_id,model_call_id),KEY idx_agent_usage_settlement(settlement_status,next_settlement_attempt_at,id),
 KEY idx_agent_usage_run_status(run_id,status,id),
 CONSTRAINT chk_agent_usage_status CHECK(status IN('STARTED','COMPLETED','FAILED','CANCELLED')),
 CONSTRAINT chk_agent_usage_settlement CHECK(settlement_status IN('PENDING','CLAIMED','SETTLED')));
```
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentPersistenceMigrationIT" verify`; expect PASS including non-null deadline plus its maintenance index, unique `(parent_run_id,parent_tool_call_id)`, one active root plus active children in the same conversation, rejection of a second active root, `next_message_order DEFAULT 1`, unique full-history message order, projection/outbox columns, stable reorder of duplicate/deleted rows, and MySQL version rejection below 8.0.16.
- [ ] Commit: `git add ai-fusion-video/pom.xml ai-fusion-video/src/main/resources/db/migration/V1.0.6.1.5__agent_run_and_event.sql ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentPersistenceMigrationIT.java; git commit -m "feat(ai): add durable runtime schema"`.

### Task 2: Map entities, mappers, and message allocator

**Files:** Modify `entity/ai/AgentConversation.java`, `AgentMessage.java`, `mapper/ai/AgentConversationMapper.java`, `AgentMessageMapper.java`, `service/ai/AgentMessageService.java`, `service/ai/AgentConversationService.java`, `service/task/TaskStreamService.java`; create `entity/ai/AgentRun.java`, `AgentEvent.java`, `AgentModelCallUsage.java`, three mappers, `enums/ai/AgentRuntimeErrorCode.java`, `service/ai/run/AgentMessageAllocator.java`, `integration/AgentMessageAllocatorIT.java`.

**Interfaces:** Produces `long AgentMessageAllocator.append(String conversationId,AgentMessage message)` and locked run/event mapper reads. `AgentRun` maps non-null `deadlineAt` plus `parentRunId`, `parentToolCallId`, and `agentName` exactly to the Task 1 columns; the run mapper exposes `selectActiveChildren(parentRunId)` only through `idx_agent_run_parent_status` and `selectByParentAndToolCallForUpdate(parentRunId,parentToolCallId)` through the unique child-admission key.

- [ ] Write a 32-thread integration test asserting unique orders `1..32`, `message_count=32`, and start/update alone does not increment count. Add mapper round-trip assertions for non-null deadline, root runs with all three parent fields NULL, and child runs with exact `parentRunId/parentToolCallId/agentName` values.
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentMessageAllocatorIT" verify`; expect FAIL from duplicate orders/current `MAX+1`.
- [ ] Implement the allocator and route user/assistant/tool/TaskStream writers through it:
```java
public enum AgentRuntimeErrorCode {
  MODEL_AUTH_FAILED, MODEL_NOT_FOUND, MODEL_PROTOCOL_ERROR, MODEL_TIMEOUT,
  TOOL_VALIDATION_FAILED, TOOL_TIMEOUT, TOOL_CANCELLED, STATE_STORE_FAILED,
  EVENT_PERSIST_FAILED, AGENT_EVENT_BACKPRESSURE_OVERFLOW, HARNESS_CAPACITY_EXHAUSTED,
  RUN_CONFIG_UNAVAILABLE, CONFIRMATION_EXPIRED, EXTERNAL_EXECUTION_EXPIRED,
  SSE_CURSOR_INVALID, OWNER_LOST, RUN_CANCELLED, AGENTSCOPE_INTERNAL_ERROR
}

@Transactional public long append(String conversationId, AgentMessage message) {
  AgentConversation c=conversationMapper.selectByConversationIdForUpdate(conversationId);
  long order=c.getNextMessageOrder(); message.setConversationId(conversationId); message.setMessageOrder(order);
  if(messageMapper.insert(message)!=1) throw new IllegalStateException("message insert failed");
  c.setNextMessageOrder(order+1); c.setMessageCount(c.getMessageCount()+1); c.setLastMessageTime(LocalDateTime.now());
  if(conversationMapper.updateById(c)!=1) throw new IllegalStateException("counter update failed");
  return order;
}
```
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentMessageAllocatorIT" verify` and `.\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentMessageServiceTests" test`; expect PASS and no `findMaxMessageOrder` reference.
- [ ] Commit with `git commit -m "fix(ai): serialize message order allocation"` including all listed files.

### Task 3: Start a run and user message atomically

**Files:** Create `service/ai/run/model/StartAgentRunCommand.java`, `StartedAgentRun.java`, `StartChildAgentRunCommand.java`, `ChildRunAdmission.java`, `ChildRunIdentityConflictException.java`, `service/ai/run/kernel/AgentKernelSnapshot.java`, `AgentKernelSnapshotPayload.java`, `ToolManifestSnapshot.java`, `AgentKernelSnapshotBuilder.java`, `CanonicalAgentKernelSnapshotBuilder.java`, `AgentKernelSnapshotResolver.java`, `PersistedAgentKernelSnapshotResolver.java`, `RunConfigUnavailableException.java`, `repository/ai/AgentRunRepository.java`, `service/ai/run/AgentRunCoordinator.java`, `test/.../run/kernel/AgentKernelSnapshotContractTests.java`, `integration/AgentRunStartIT.java`; modify the dependency/kernel slice's `service/ai/agentscope/kernel/AgentKernelSpec.java` only to expose the immutable values serialized below.

**Interfaces:** Produces `Mono<StartedAgentRun> start(StartAgentRunCommand)` for roots, `Mono<ChildRunAdmission> startChild(StartChildAgentRunCommand)` for the only child-admission transaction, the frozen snapshot builder/resolver, and an immutable schema-v1 snapshot. Root commands set all parent fields NULL; platform child commands set all three parent identity fields. Every command/result carries the persisted deadline. Snapshot resolution verifies the SHA-256 fingerprint and exact model/tool implementation compatibility; any missing version or incompatible manifest raises `RunConfigUnavailableException` carrying `RUN_CONFIG_UNAVAILABLE`.

- [ ] Write tests for rollback on message failure, cross-user conversation rejection, generated active-slot 409, required future root deadline, child deadline not exceeding its parent, root/child parent identity, canonical snapshot hashing, secret rejection, tampered snapshot rejection, missing model version, and incompatible tool manifest.
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentRunStartIT" verify`; expect FAIL because no run transaction exists.
- [ ] Freeze the persisted snapshot and start command. `snapshotJson` contains no API key, Authorization, proxy password, raw Base64, signed query, or mutable latest-version lookup:
```java
public record ToolManifestSnapshot(
    String name,
    String schemaSha256,
    boolean readOnly,
    boolean concurrencySafe,
    String implementationVersion) {}

public record AgentKernelSnapshotPayload(
    int schemaVersion,
    String agentDefinitionStableKey,
    String agentName,
    String description,
    String systemPrompt,
    int maxIters,
    String modelConfigId,
    long modelConfigVersion,
    String provider,
    String modelCode,
    JsonNode modelOptions,
    List<ToolManifestSnapshot> tools,
    String applicationVersion) {
  public AgentKernelSnapshotPayload {
    if (schemaVersion != 1) throw new IllegalArgumentException("unsupported kernel snapshot schema");
    tools = List.copyOf(tools).stream().sorted(Comparator.comparing(ToolManifestSnapshot::name)).toList();
  }
}

public record StartAgentRunCommand(
    String runId, String conversationId, long userId, Long projectId, String agentType,
    String parentRunId, String parentToolCallId, String agentName,
    String agentStateSessionId, AgentKernelSnapshot kernelSnapshot,
    String ownerInstanceId, Duration ownerLease,
    Instant deadline, String userContent, String referencesJson) {}

public record StartedAgentRun(
    String runId, String conversationId, String agentStateSessionId,
    String ownerInstanceId, long ownerEpoch, Instant leaseUntil,
    Instant deadline, AgentKernelSnapshot kernelSnapshot, long initialMessageOrder) {}

public record StartChildAgentRunCommand(
    String childRunId,
    String parentRunId,
    String parentToolCallId,
    String parentOwnerInstanceId,
    long parentOwnerEpoch,
    String agentName,
    String agentDefinitionStableKey,
    AgentKernelSnapshot kernelSnapshot,
    String ownerInstanceId,
    Duration ownerLease,
    Instant deadline,
    String userContent,
    String referencesJson) {}

public record ChildRunAdmission(
    StartedAgentRun run,
    AgentRunStatus status,
    boolean created) {}
```
- [ ] Implement canonical build/resolve. Builder serializes `AgentKernelSnapshotPayload` with sorted object keys and tool names, hashes the exact UTF-8 bytes, and rejects the forbidden secret-field/value patterns. Resolver parses schema 1, recomputes the hash with `MessageDigest.isEqual`, requires the exact persisted model config version and every tool `implementationVersion/schemaSha256`, and maps every absence or incompatibility to `RunConfigUnavailableException(RUN_CONFIG_UNAVAILABLE)`.
- [ ] Implement one short transaction:
```java
@Transactional public StartedAgentRun start(StartAgentRunCommand c) {
  AgentConversation conv=requireOwnedConversation(c.conversationId(),c.userId()); LocalDateTime now=runMapper.selectDatabaseNow();
  requireFuture(c.deadline(),now);
  AgentRun run=AgentRun.running(c.runId(),conv.getConversationId(),c.userId(),c.agentStateSessionId(),
      c.kernelSnapshot().fingerprint(),c.kernelSnapshot().snapshotJson(),c.ownerInstanceId(),1L,
      c.parentRunId(),c.parentToolCallId(),c.agentName(),toLocal(c.deadline()),now,now.plus(c.ownerLease()));
  try { runMapper.insert(run); } catch(DuplicateKeyException e) { throw new BusinessException("RUN_ALREADY_ACTIVE","该会话已有运行中的任务"); }
  long order=allocator.append(c.conversationId(),AgentMessage.user(c.runId(),c.userContent(),c.referencesJson()));
  return new StartedAgentRun(c.runId(),c.conversationId(),c.agentStateSessionId(),
      c.ownerInstanceId(),1L,toInstant(run.getLeaseUntil()),c.deadline(),c.kernelSnapshot(),order);
}
```
- [ ] Implement `startChild` as one parent-row-locked transaction. It is the only child insert path; root `start` rejects non-null parent fields:
```java
@Transactional public ChildRunAdmission startChildTx(StartChildAgentRunCommand command) {
  LocalDateTime now=runMapper.selectDatabaseNow();
  AgentRun parent=runMapper.selectByRunIdForUpdate(command.parentRunId());
  requireState(parent,RUNNING);
  requireOwner(parent,command.parentOwnerInstanceId(),command.parentOwnerEpoch());
  requireLeaseAfter(parent,now);
  requireFuture(command.deadline(),now);
  if(command.deadline().isAfter(toInstant(parent.getDeadlineAt()))) throw childDeadlineAfterParent();
  try {
    AgentRun child=insertChildRunAndUserMessage(parent,command,now);
    return new ChildRunAdmission(toStarted(child),RUNNING,true);
  } catch (DuplicateKeyException duplicate) {
    AgentRun existing=runMapper.selectByParentAndToolCallForUpdate(
        command.parentRunId(),command.parentToolCallId());
    if(existing==null) throw duplicate;
    if(!Objects.equals(existing.getAgentName(),command.agentName())
        || !Objects.equals(existing.getKernelFingerprint(),command.kernelSnapshot().fingerprint())
        || !Objects.equals(existing.getAgentDefinitionSnapshotJson(),command.kernelSnapshot().snapshotJson())
        || !Objects.equals(existing.getDeadlineAt(),toLocal(command.deadline()))) {
      throw new ChildRunIdentityConflictException(command.parentRunId(),command.parentToolCallId());
    }
    return new ChildRunAdmission(toStarted(existing),existing.getStatus(),false);
  }
}
```
`insertChildRunAndUserMessage` derives conversation/user/project and the child session from the locked parent plus `agentDefinitionStableKey`; callers cannot supply those authoritative parent values. The matching duplicate returns the existing child and status without appending another user message or launching another execution. A mismatch in agent name, snapshot fingerprint/bytes, or deadline fails closed. Parent cancellation uses the same parent row lock before its descendant CAS, so either child insertion commits first and is cancelled, or cancellation commits first and child admission rejects.
- [ ] Run `.\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentKernelSnapshotContractTests" test` and `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentRunStartIT" verify`; expect PASS with byte-stable snapshots, every unavailable/incompatible resolution reported as `RUN_CONFIG_UNAVAILABLE`, exact child identity/deadline persistence, and no RUNNING row after a forced user-message failure.
- [ ] Commit with `git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/StartAgentRunCommand.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/StartedAgentRun.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/StartChildAgentRunCommand.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/ChildRunAdmission.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/ChildRunIdentityConflictException.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/AgentKernelSnapshot.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/AgentKernelSnapshotPayload.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/ToolManifestSnapshot.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/AgentKernelSnapshotBuilder.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/CanonicalAgentKernelSnapshotBuilder.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/AgentKernelSnapshotResolver.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/PersistedAgentKernelSnapshotResolver.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/kernel/RunConfigUnavailableException.java ai-fusion-video/src/main/java/com/stonewu/fusion/repository/ai/AgentRunRepository.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRunCoordinator.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/AgentKernelSpec.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/kernel/AgentKernelSnapshotContractTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentRunStartIT.java; git commit -m "feat(ai): start agent runs atomically"`.

### Task 4: Journal per-run sequence and terminal CAS

**Files:** Create `service/ai/run/model/AgentEventEnvelope.java`, `CommittedAgentEvent.java`, `RunTerminalRequest.java`, `SystemTerminalActor.java`, `service/ai/run/AgentScopeEventMapper.java`, `AgentEventEnvelopeSanitizer.java`, `repository/ai/AgentEventRepository.java`, `service/ai/run/AgentEventJournal.java`, `MySqlAgentEventJournal.java`, `RunTerminalCoordinator.java`, `MySqlRunTerminalCoordinator.java`, `test/.../run/AgentScopeEventMapperTests.java`, `integration/AgentJournalTerminalIT.java`.

**Interfaces:** Produces exhaustive `AgentEventEnvelope map(AgentEvent)` plus exactly the frozen `appendOwned`, `terminateOwned`, and `terminateSystem` methods. Every ordinary append and owner-originated terminal carries `ownerInstanceId/ownerEpoch`; while holding the run row lock, the transaction verifies both values, `status=RUNNING`, and `lease_until > databaseNow`. A mismatch returns `Optional.empty()` and changes neither run nor event. `terminateSystem` has no owner bypass flag: its actor enum has only cancellation and reconciler, and it rejects every transition outside `CANCEL_REQUESTED -> CANCELLED`, expired `RUNNING -> FAILED/OWNER_LOST`, or expired `CANCEL_REQUESTED -> CANCELLED`. `RunTerminalRequest` carries the run's `StateStoreSlot`; a requested `COMPLETED` transition is converted to `FAILED/STATE_STORE_FAILED` before the transaction if the fail-closed guard reports a save failure.

- [ ] Write the event completeness test against `AgentEventType.values()` and a fixture map containing every concrete subtype; assert exactly 31 values, all identity fields are subtype-derived, and sanitizer output contains no Authorization, credential, raw Base64, binary, signed-query secret, or uncontrolled file path.
- [ ] Implement an exhaustive switch with no default branch so a future GA enum addition breaks compilation or the completeness test:
```java
return switch (event.getType()) {
  case AGENT_START, AGENT_END, AGENT_RESULT -> mapAgentLifecycle(event);
  case MODEL_CALL_START, MODEL_CALL_END -> mapModelCall(event);
  case TEXT_BLOCK_START, TEXT_BLOCK_DELTA, TEXT_BLOCK_END -> mapText(event);
  case THINKING_BLOCK_START, THINKING_BLOCK_DELTA, THINKING_BLOCK_END -> mapThinking(event);
  case DATA_BLOCK_START, DATA_BLOCK_DELTA, DATA_BLOCK_END -> mapData(event);
  case TOOL_CALL_START, TOOL_CALL_DELTA, TOOL_CALL_END -> mapToolCall(event);
  case TOOL_RESULT_START, TOOL_RESULT_TEXT_DELTA, TOOL_RESULT_DATA_DELTA, TOOL_RESULT_END -> mapToolResult(event);
  case EXCEED_MAX_ITERS, REQUEST_STOP, ALL_TOOLS_DENIED -> mapControl(event);
  case REQUIRE_USER_CONFIRM, REQUIRE_EXTERNAL_EXECUTION,
       USER_CONFIRM_RESULT, EXTERNAL_EXECUTION_RESULT -> mapWaiting(event);
  case SUBAGENT_EXPOSED, HINT_BLOCK, CUSTOM -> mapExtension(event);
};
```
- [ ] Run `.\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentScopeEventMapperTests" test`; expect PASS only when all 31 GA enum constants, raw-only lifecycle events, legacy projection fields, `CUSTOM`, and main-terminal null `parentToolCallId/agentName` contracts are covered.

- [ ] Write concurrent append and complete/fail/cancel race tests asserting sequence `1..N`, one terminal row, and no event after terminal. Add owner mismatch, epoch mismatch, lease-equal-to-database-now, expired lease, and save-failure tests. For `terminateSystem`, assert `CANCELLATION_COORDINATOR` accepts only locked `CANCEL_REQUESTED -> CANCELLED`; assert `OWNER_RECONCILER` returns empty for non-expired `RUNNING` and non-expired `CANCEL_REQUESTED`, and only accepts expired `RUNNING -> FAILED/OWNER_LOST` or expired `CANCEL_REQUESTED -> CANCELLED` using database time. Assert each rejected call affects zero run/event rows; assert a requested `COMPLETED` with a recorded `StateStoreFailure` produces one `FAILED/STATE_STORE_FAILED` terminal instead.
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentJournalTerminalIT" verify`; expect FAIL because sequence/terminal transactions are absent.
- [ ] Implement locked append and terminal paths:
```java
@Transactional Optional<CommittedAgentEvent> appendOwnedTx(String runId,String ownerInstanceId,long ownerEpoch,AgentEventEnvelope e) {
  AgentRun run=runMapper.selectByRunIdForUpdate(runId); LocalDateTime now=runMapper.selectDatabaseNow();
  if(!isCurrentOwner(run,ownerInstanceId,ownerEpoch,now)) return Optional.empty();
  long seq=run.getNextSequence(); run.setNextSequence(seq+1); runMapper.updateById(run);
  AgentEvent row=AgentEvent.from(runId,seq,e,e.outputType()==null?NOT_REQUIRED:PENDING); eventMapper.insert(row);
  return Optional.of(toCommitted(row));
}

private boolean isCurrentOwner(AgentRun run,String ownerInstanceId,long ownerEpoch,LocalDateTime databaseNow) {
  return run.getStatus()==RUNNING
      && Objects.equals(run.getOwnerInstanceId(),ownerInstanceId)
      && run.getOwnerEpoch()==ownerEpoch
      && run.getLeaseUntil()!=null
      && run.getLeaseUntil().isAfter(databaseNow);
}

public Mono<Optional<CommittedAgentEvent>> appendOwned(String runId,String owner,long epoch,AgentEventEnvelope event) {
  return Mono.fromCallable(()->transactions.appendOwnedTx(runId,owner,epoch,event)).subscribeOn(schedulers.journal());
}

public Mono<Optional<CommittedAgentEvent>> terminateOwned(RunTerminalRequest request,String owner,long epoch) {
  RunTerminalRequest guarded=failClosedCompletion(request);
  return Mono.fromCallable(()->transactions.terminateOwnedTx(guarded,owner,epoch)).subscribeOn(schedulers.journal());
}

public Mono<Optional<CommittedAgentEvent>> terminateSystem(RunTerminalRequest request,SystemTerminalActor actor) {
  return Mono.fromCallable(()->transactions.terminateSystemTx(request,actor)).subscribeOn(schedulers.journal());
}

private RunTerminalRequest failClosedCompletion(RunTerminalRequest r) {
  if(r.terminalStatus()==COMPLETED){
    try{stateStoreFailureGuard.throwIfFailed(r.stateStoreSlot());}
    catch(StateStoreFailure failure){return r.asFailure(STATE_STORE_FAILED,sanitize(failure));}
  }
  return r;
}

@Transactional Optional<CommittedAgentEvent> terminateOwnedTx(RunTerminalRequest r,String owner,long epoch) {
  AgentRun run=runMapper.selectByRunIdForUpdate(r.runId()); LocalDateTime now=runMapper.selectDatabaseNow();
  if(!isCurrentOwner(run,owner,epoch,now)||!r.expectedStatuses().equals(Set.of(RUNNING))) return Optional.empty();
  return insertTerminalAndFinish(run,r,now);
}

@Transactional Optional<CommittedAgentEvent> terminateSystemTx(RunTerminalRequest r,SystemTerminalActor actor) {
  AgentRun run=runMapper.selectByRunIdForUpdate(r.runId()); LocalDateTime now=runMapper.selectDatabaseNow();
  boolean allowed=switch(actor){
    case CANCELLATION_COORDINATOR -> run.getStatus()==CANCEL_REQUESTED && r.terminalStatus()==CANCELLED;
    case OWNER_RECONCILER -> run.getLeaseUntil()!=null && !run.getLeaseUntil().isAfter(now)
        && ((run.getStatus()==RUNNING && r.terminalStatus()==FAILED && r.errorCode()==OWNER_LOST)
            || (run.getStatus()==CANCEL_REQUESTED && r.terminalStatus()==CANCELLED));
  };
  if(!allowed) return Optional.empty();
  return insertTerminalAndFinish(run,r,now);
}

private Optional<CommittedAgentEvent> insertTerminalAndFinish(AgentRun run,RunTerminalRequest r,LocalDateTime now) {
  long seq=run.getNextSequence(); eventMapper.insert(AgentEvent.terminal(r.runId(),seq,r.terminalEnvelope()));
  run.finish(r.terminalStatus(),seq,r.terminalEnvelope().outputType(),r.errorCode(),r.errorMessage(),now);
  runMapper.updateById(run); return Optional.of(toCommitted(eventMapper.selectByRunAndSequence(r.runId(),seq)));
}
```
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentJournalTerminalIT" verify`; expect PASS with exactly one terminal under 100 repeated races, no stale-owner writes, and no production reference to `terminateSystem` outside `CancellationCoordinator` and `AgentRunReconciliationService`.
- [ ] Commit with `git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/AgentEventEnvelope.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/CommittedAgentEvent.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/RunTerminalRequest.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/SystemTerminalActor.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentScopeEventMapper.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentEventEnvelopeSanitizer.java ai-fusion-video/src/main/java/com/stonewu/fusion/repository/ai/AgentEventRepository.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentEventJournal.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/MySqlAgentEventJournal.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/RunTerminalCoordinator.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/MySqlRunTerminalCoordinator.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/AgentScopeEventMapperTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentJournalTerminalIT.java; git commit -m "feat(ai): journal events with terminal CAS"`.

### Task 5: Persist model calls and settlement claims

**Files:** Create `repository/ai/AgentModelCallUsageRepository.java`, `repository/ai/MybatisAgentModelCallUsageRepository.java`, `service/ai/run/model/NormalizedModelUsage.java`, `service/ai/run/model/ModelCallIdentityConflictException.java`, `service/ai/run/ModelUsageSettlementPort.java`, `service/ai/run/AuditLedgerModelUsageSettlementAdapter.java`, `config/AgentUsageSettlementConfiguration.java`, `test/.../run/AuditLedgerModelUsageSettlementAdapterTests.java`, `test/.../config/AgentUsageSettlementConfigurationTests.java`, `integration/AgentModelCallUsageIT.java`.

**Interfaces:** Implements the frozen `AgentModelCallUsageRepository` and owns the frozen `ModelUsageSettlementPort`, its production audit-ledger adapter, and the single Spring binding. The later model/tool plan owns `ModelCallUsageSettlementWorker` only and must not recreate the port or binding. Because this product has no billing subsystem, the production adapter's durable effect is the already-persisted usage ledger and it returns a deterministic settlement ID; a future billing adapter replaces it with `@ConditionalOnMissingBean` without changing the worker.

- [ ] Write tests for two calls in one run; concurrent duplicate `startCall` with identical `(runId,modelCallId,provider,modelCode)` succeeding idempotently; the same call identity with a different provider or model failing closed; disjoint claims; owner-guarded settle/retry; deterministic settlement IDs; exactly one Spring `ModelUsageSettlementPort`; and run settled only when the locked run is terminal and no unsettled call exists. Add the ordering regression: while the run is `RUNNING`, settle call 1 and assert `markRunUsageSettledIfAllCallsSettled` returns false, then start call 2 successfully; after terminalization the flag remains false until call 2 is settled.
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentModelCallUsageIT" verify`; expect FAIL because usage repository is absent.
- [ ] Implement claim/complete operations:
```java
@Transactional public List<AgentModelCallUsage> claimSettlementBatch(String owner,Duration lease,int limit){
  List<AgentModelCallUsage> rows=mapper.selectCandidatesForUpdate(limit); LocalDateTime now=runMapper.selectDatabaseNow();
  rows.forEach(r->{r.setSettlementStatus(CLAIMED);r.setSettlementClaimOwner(owner);r.setSettlementClaimUntil(now.plus(lease));r.setSettlementAttempts(r.getSettlementAttempts()+1);mapper.updateById(r);});
  return List.copyOf(rows);
}
@Transactional public void startCall(String runId,String callId,String provider,String model){
  AgentModelCallUsage started=AgentModelCallUsage.started(runId,callId,provider,model,runMapper.selectDatabaseNow());
  try{mapper.insert(started);return;}
  catch(DuplicateKeyException duplicate){
    AgentModelCallUsage existing=mapper.selectByRunAndCall(runId,callId);
    if(existing==null) throw duplicate;
    if(Objects.equals(existing.getProvider(),provider)&&Objects.equals(existing.getModelCode(),model)) return;
    throw new ModelCallIdentityConflictException(runId,callId,
        existing.getProvider(),existing.getModelCode(),provider,model);
  }
}
public boolean completeCall(String runId,String callId,NormalizedModelUsage usage){return mapper.completeStarted(runId,callId,usage)==1;}
public boolean failCall(String runId,String callId,AgentModelCallStatus status){return mapper.finishStarted(runId,callId,status)==1;}
public boolean markSettled(long id,String owner,String downstreamId){return mapper.markSettled(id,owner,downstreamId)==1;}
public boolean releaseSettlementForRetry(long id,String owner,Instant at,String error){return mapper.releaseRetry(id,owner,at,error)==1;}
@Transactional public boolean markRunUsageSettledIfAllCallsSettled(String runId){
  AgentRun run=runMapper.selectByRunIdForUpdate(runId);
  if(!Set.of(COMPLETED,FAILED,CANCELLED).contains(run.getStatus())) return false;
  return runMapper.markUsageSettledIfNoUnsettledCall(runId,runMapper.selectDatabaseNow())==1;
}
```
`markUsageSettledIfNoUnsettledCall` is one guarded statement; the prior `countUnsettled` then update sequence is forbidden:
```sql
UPDATE afv_agent_run r
 SET r.usage_settled=1,r.usage_settled_at=?,r.update_time=CURRENT_TIMESTAMP(3)
 WHERE r.run_id=? AND r.usage_settled=0
   AND r.status IN('COMPLETED','FAILED','CANCELLED')
   AND NOT EXISTS (
     SELECT 1 FROM afv_agent_model_call_usage u
      WHERE u.run_id=r.run_id AND u.settlement_status<>'SETTLED');
```
- [ ] Implement and bind the production settlement adapter:
```java
public final class AuditLedgerModelUsageSettlementAdapter implements ModelUsageSettlementPort {
  @Override public Mono<String> settle(String idempotencyKey,NormalizedModelUsage usage){
    if(idempotencyKey==null||!idempotencyKey.matches("[^:]+:[^:]+"))
      return Mono.error(new IllegalArgumentException("idempotency key must be runId:modelCallId"));
    Objects.requireNonNull(usage,"usage");
    return Mono.just("usage-"+HexFormat.of().formatHex(
        sha256(("afv-usage-v1:"+idempotencyKey).getBytes(StandardCharsets.UTF_8))));
  }
  private static byte[] sha256(byte[] value){
    try{return MessageDigest.getInstance("SHA-256").digest(value);}
    catch(NoSuchAlgorithmException impossible){throw new IllegalStateException(impossible);}
  }
}

@Configuration
public class AgentUsageSettlementConfiguration {
  @Bean
  @ConditionalOnMissingBean(ModelUsageSettlementPort.class)
  ModelUsageSettlementPort modelUsageSettlementPort(){
    return new AuditLedgerModelUsageSettlementAdapter();
  }
}
```
- [ ] Run `.\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AuditLedgerModelUsageSettlementAdapterTests,AgentUsageSettlementConfigurationTests" test` and `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentModelCallUsageIT" verify`; expect PASS, same provider/model duplicate starts to remain one row, conflicting identity to throw `ModelCallIdentityConflictException`, one production port bean, stale claim-owner updates to affect zero rows, a RUNNING run never marked usage-settled after call 1, and terminal usage marked only after call 2 settles.
- [ ] Commit with `git add ai-fusion-video/src/main/java/com/stonewu/fusion/repository/ai ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/ModelUsageSettlementPort.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AuditLedgerModelUsageSettlementAdapter.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model ai-fusion-video/src/main/java/com/stonewu/fusion/config/AgentUsageSettlementConfiguration.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentModelCallUsageIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/AuditLedgerModelUsageSettlementAdapterTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/config/AgentUsageSettlementConfigurationTests.java; git commit -m "feat(ai): persist and bind model usage settlement"`.

### Task 6: Publish committed events through a durable outbox

**Files:** Create `service/ai/run/AgentRunRedisSignalService.java`, `AgentEventOutboxPublisher.java`, `integration/AgentOutboxMultiInstanceIT.java`; modify `service/ai/AiStreamRedisService.java`.

**Interfaces:** Produces `Mono<Void> publishBatch(String owner,int limit)` and `Flux<Long> wakeups(String runId)`; Redis payload is wake-up only.

- [ ] Write two-node tests for disjoint `FOR UPDATE SKIP LOCKED` claims, expired claim takeover, duplicate publish, and poison-event backoff.
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentOutboxMultiInstanceIT" verify`; expect FAIL because current Redis stream is the source of truth.
- [ ] Implement short claim, transaction-free Redis, and owner-guarded result:
```java
return claim(owner,limit).flatMapMany(Flux::fromIterable).concatMap(e->signals.publishWakeup(e.getRunId(),e.getSequenceNo())
  .then(markPublished(e.getId(),owner)).onErrorResume(x->releaseRetry(e.getId(),owner,jitter(e.getPublishAttempts()),sanitize(x)))).then();
```
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentOutboxMultiInstanceIT" verify`; expect PASS, no shared claims, expired takeover, and retry delay between 100ms and 30s.
- [ ] Commit with `git commit -m "feat(ai): publish events through durable outbox"`.

### Task 7: Project event history idempotently

**Files:** Create `service/ai/run/AgentMessageProjectionService.java`, `integration/AgentProjectionRecoveryIT.java`; modify `AgentEventMapper.java`, `AgentMessageService.java`.

**Interfaces:** Produces `Mono<Void> projectThrough(String runId,long throughSequence)`.

- [ ] Write crash-after-insert retry tests asserting unique SHA-256 projection keys and terminal projection completion.
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentProjectionRecoveryIT" verify`; expect FAIL because current projection swallows persistence exceptions.
- [ ] Implement locked cursor advancement:
```java
String key=sha256(runId+":"+event.getSequenceNo()+":"+kind);
message.setRunId(runId);message.setProjectionKey(key);
AgentMessage existing=messageMapper.selectByProjectionKey(key);
if(existing==null){allocator.append(run.getConversationId(),message);}else{requireSameProjection(existing,runId,event.getSequenceNo(),kind);}
run.setProjectedThroughSequence(event.getSequenceNo());
if(run.getTerminalSequence()!=null&&run.getProjectedThroughSequence()>=run.getTerminalSequence())run.setProjectionCompletedAt(dbNow());
```
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentProjectionRecoveryIT" verify`; expect PASS with one message per projection key and resumable cursor.
- [ ] Commit with `git commit -m "feat(ai): project committed agent events"`.

### Task 8: Bound ingress and own execution server-side

**Files:** Create `service/ai/run/BoundedAgentEventIngress.java`, `AgentEventChunkCoalescer.java`, `AgentExecution.java`, `AgentExecutionHandle.java`, `OwnedExecutionRegistry.java`, `AgentExecutionFactory.java`, `AgentExecutionRuntimeContextRequests.java`, `AgentInputHistoryMapper.java`, `RunShutdownCancellationPort.java`, `RunExecutionSupervisor.java`, `DefaultRunExecutionSupervisor.java`, `PlatformSubAgentRunService.java`, `service/ai/run/model/StartAgentExecutionCommand.java`, `ResumeAgentExecutionCommand.java`, `ExecutionStopReason.java`, `service/ai/agentscope/tool/PlatformSubAgentCommand.java`, `PlatformSubAgentRun.java`, `PlatformSubAgentRunPort.java`, `test/.../run/BoundedAgentEventIngressTests.java`, `AgentEventChunkCoalescerTests.java`, `RunExecutionSupervisorTests.java`, `integration/PlatformSubAgentRunServiceIT.java`; modify `repository/ai/AgentRunRepository.java`, `service/ai/run/AgentRunCoordinator.java`, `ai-fusion-video/src/main/resources/application.yaml`, and `service/ai/agentscope/AgentScopeAssistantService.java`; consume, do not recreate, the dependency/kernel plan's `service/ai/run/AgentRuntimeShutdownPort.java` and `service/ai/agentscope/runtime/AgentRuntimeSchedulers.java`. Durable Runtime owns every Start/Resume/ExecutionStopReason/Supervisor file listed here and creates no shadow under `service/ai/agentscope/runtime`.

**Interfaces:** Creates `RunExecutionSupervisor extends AgentRuntimeShutdownPort` with the frozen `start(StartAgentExecutionCommand)`, `resume(ResumeAgentExecutionCommand)`, `interruptOwned(...)`, and inherited `shutdown(Duration)` signatures at `service/ai/run`. Start receives the exact input messages, persisted Kernel snapshot, live `AgentKernelSpec`, complete `AgentScopeRuntimeContextRequest`, and the persisted deadline through `StartedAgentRun`; resume receives messages, snapshot, new-owner RuntimeContext input, and persisted deadline through `ResumedAgentRun`, then resolves the spec itself. It also provides the production `PlatformSubAgentRunService implements PlatformSubAgentRunPort`. Execution handles and subscriptions are server-owned before `start`/`resume` completes; max ingress is `4096` events or `8MiB` per run. `AgentEventChunkCoalescer.coalesce(Flux<AgentEventEnvelope>)` merges only contiguous text or thinking deltas up to `50ms` or `1024` characters and flushes immediately on block end, tool, error, cancel, or terminal boundaries.

- [ ] Write tests proving observer disposal does not interrupt execution, `start`/`resume` return only after a registry-owned subscription is launched, overflow writes `AGENT_EVENT_BACKPRESSURE_OVERFLOW`, deadline expiry interrupts with `DEADLINE`, and close occurs once. Assert start forwards the exact messages/spec/snapshot/RuntimeContext/deadline input; resume resolves the persisted snapshot, uses the new owner tuple/deadline, and maps `RunConfigUnavailableException` through `terminateOwned(...FAILED/RUN_CONFIG_UNAVAILABLE...)`. Assert `interruptOwned` ignores a stale epoch and `shutdown` rejects new starts, drains for the supplied duration, persists shutdown cancellation for survivors, calls the kernel lease-cache drain through its inherited shutdown contract, and closes each handle once. With `VirtualTimeScheduler`, assert text/thinking chunks flush at 50ms or 1024 characters, never cross reply/block/source identity, and flush before tool/error/cancel/terminal events.
- [ ] Write `PlatformSubAgentRunServiceIT` proving the locked parent must be `RUNNING`, owned by exact `parentOwnerInstanceId/parentOwnerEpoch`, have `lease_until > databaseNow`, and have a deadline no earlier than the child. Assert first admission persists `parent_run_id/parent_tool_call_id/agent_name/deadline_at`, uses an independent session and child runId, passes a child-scoped `StartAgentExecutionCommand` once, returns status, and returns no native AgentScope taskId. Retry the same parent/tool/agent/snapshot and assert the same child/status returns without a second message/execution; retry with a changed agent or snapshot and assert `ChildRunIdentityConflictException`. Add two-node takeover rejection and a concurrent parent-cancel-vs-child-start test proving no active child can appear beneath a cancelled parent. Assert `cancelChildren(parentRunId)` locks the parent row before CASing every active descendant and leaves unrelated/root runs unchanged.
- [ ] Run `.\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=BoundedAgentEventIngressTests,AgentEventChunkCoalescerTests,RunExecutionSupervisorTests" test` and `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=PlatformSubAgentRunServiceIT" verify`; expect FAIL because execution is HTTP/local-map owned and the durable child-run port has no production implementation.
- [ ] Implement non-blocking queue admission and supervisor:
```java
boolean offer(AgentEventEnvelope e){long b=utf8Size(e.payloadJson());long n=bytes.addAndGet(b);if(n>maxBytes||!queue.offer(e)){bytes.addAndGet(-b);return false;}return true;}
public Mono<Void> start(StartAgentExecutionCommand c){
  requireMatchingSnapshot(c.kernelSnapshot(),c.kernelSpec());
  return startResolved(c.run().runId(),c.run().ownerInstanceId(),c.run().ownerEpoch(),
      c.run().deadline(),c.messages(),c.kernelSpec(),c.runtimeContextRequest());
}

public Mono<Void> resume(ResumeAgentExecutionCommand c){
  return snapshots.resolve(c.kernelSnapshot())
      .flatMap(spec->startResolved(c.run().runId(),c.run().newOwnerInstanceId(),c.run().newOwnerEpoch(),
          c.run().deadline(),c.messages(),spec,c.runtimeContextRequest()))
      .onErrorResume(RunConfigUnavailableException.class,failure->
          terminals.terminateOwned(RunTerminalRequest.failed(c.run().runId(),RUN_CONFIG_UNAVAILABLE,
              sanitizer.message(failure)),c.run().newOwnerInstanceId(),c.run().newOwnerEpoch()).then());
}

private Mono<Void> startResolved(String runId,String owner,long epoch,Instant deadline,List<Msg> messages,
    AgentKernelSpec spec,AgentScopeRuntimeContextRequest runtimeRequest){
  requireMatchingDeadline(runtimeRequest,deadline);
  return factory.start(runId,owner,epoch,messages,spec,runtimeRequest)
      .flatMap(execution->executions.registerAndLaunch(execution,deadline,
          events->chunks.coalesce(events)
              .concatMap(event->journal.appendOwned(runId,owner,epoch,event))));
}

public Mono<Boolean> interruptOwned(String runId,String owner,long epoch,ExecutionStopReason reason){
  return executions.interruptOwned(runId,owner,epoch,reason);
}

@Override public Mono<Void> shutdown(Duration drainTimeout){
  accepting.set(false);
  return awaitEmpty(drainTimeout)
      .thenMany(Flux.fromIterable(executions.snapshot()))
      .concatMap(handle->shutdownCancellation.request(handle.runId()).then(handle.interrupt(SHUTDOWN)))
      .then(kernelLeaseCache.drainAndClose(drainTimeout));
}

public record ChunkPolicy(Duration maxDelay, int maxChars) {
  public static ChunkPolicy productionDefault(){return new ChunkPolicy(Duration.ofMillis(50),1024);}
}
public Flux<AgentEventEnvelope> coalesce(Flux<AgentEventEnvelope> source){
  return new IdentityPreservingChunkFlux(source, productionPolicy, timerScheduler);
}
```
- [ ] Implement durable platform child starts and descendant cancellation without an in-memory-only parent relation:
```java
@Override public Mono<PlatformSubAgentRun> start(PlatformSubAgentCommand c){
  AgentKernelSnapshot snapshot=snapshotBuilder.build(c.kernelSpec());
  StartChildAgentRunCommand admission=new StartChildAgentRunCommand(
      UUID.randomUUID().toString(),c.parentRunId(),c.parentToolCallId(),
      c.parentOwnerInstanceId(),c.parentOwnerEpoch(),c.agentName(),
      c.kernelSpec().agentDefinitionStableKey(),snapshot,instanceId,ownerLease,
      c.deadline(),inputHistory.userContent(c.messages()),null);
  return coordinator.startChild(admission).flatMap(result->{
    StartedAgentRun started=result.run();
    PlatformSubAgentRun response=new PlatformSubAgentRun(started.runId(),c.parentRunId(),
        c.parentToolCallId(),c.agentName(),result.status());
    if(!result.created()) return Mono.just(response);
    AgentScopeRuntimeContextRequest runtime=runtimeContextRequests.forChild(
        started,c.projectContext(),started.deadline());
    return supervisor.start(new StartAgentExecutionCommand(
        started,c.messages(),snapshot,c.kernelSpec(),runtime)).thenReturn(response);
  });
}

@Override public Mono<Void> cancelChildren(String parentRunId){
  return Mono.fromCallable(()->runRepository.lockParentAndRequestCancelActiveDescendants(parentRunId))
      .subscribeOn(schedulers.journal())
      .flatMapMany(Flux::fromIterable)
      .concatMap(child->signals.publishCancel(child.runId())
          .then(executions.interruptOwned(child.runId(),child.ownerInstanceId(),
              child.ownerEpoch(),CANCEL_REQUESTED)).then())
      .then();
}
```
- [ ] Run `.\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=BoundedAgentEventIngressTests,AgentEventChunkCoalescerTests,RunExecutionSupervisorTests" test` and `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=PlatformSubAgentRunServiceIT" verify`; expect PASS, server-owned execution survives observer disposal, exact command/deadline forwarding, `RUN_CONFIG_UNAVAILABLE` owned terminal mapping, idempotent and fenced child admission, cancel-vs-start serialization, persistent child deadline/status/cancellation, and no `.block()` or unbounded `Sinks` in the supervisor.
- [ ] Commit with `git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run ai-fusion-video/src/main/java/com/stonewu/fusion/repository/ai/AgentRunRepository.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformSubAgentCommand.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformSubAgentRun.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/tool/PlatformSubAgentRunPort.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/AgentScopeAssistantService.java ai-fusion-video/src/main/resources/application.yaml ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run ai-fusion-video/src/test/java/com/stonewu/fusion/integration/PlatformSubAgentRunServiceIT.java; git commit -m "feat(ai): supervise durable parent and child runs"`.

### Task 9: Implement recoverable WAITING state port

**Files:** Create `service/ai/run/AgentWaitingStatePort.java`, `DurableAgentWaitingStateService.java`, `service/ai/run/model/PendingConfirmation.java`, `PendingExternalExecution.java`, `WaitingCheckpoint.java`, `ResumeConfirmationCommand.java`, `ResumeExternalCommand.java`, `ResumedAgentRun.java`, and `integration/AgentWaitingStateIT.java`; modify run/event mappers. Consume, do not recreate, Task 3's `service/ai/run/kernel/AgentKernelSnapshotBuilder.java`, `AgentKernelSnapshotResolver.java`, and `RunConfigUnavailableException.java`.

**Interfaces:** Implements every frozen WAITING method and returns `ResumedAgentRun` with incremented owner epoch plus the exact persisted `agent_state_session_id/kernel_fingerprint/agent_definition_snapshot_json/deadline_at`. The WAITING transaction never changes the deadline or resolves against latest configuration. After the CAS returns RUNNING, `RunExecutionSupervisor.resume(ResumeAgentExecutionCommand)` owns snapshot resolution and maps `RunConfigUnavailableException` to an owner-validated `FAILED/RUN_CONFIG_UNAVAILABLE` terminal. Controller request mapping and UI behavior belong to the later cutover plan, not this task.

- [ ] Write tests for candidate-before-save invisibility, owner/epoch CAS, expiry, cross-user denial, exact decision set, external tool identity, persisted snapshot/session/deadline round-trip, cross-node resume, tampered/unavailable snapshot fail-closed mapping, and proof that no latest Agent/model/tool configuration lookup occurs before returning `ResumedAgentRun`.
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentWaitingStateIT" verify`; expect FAIL because WAITING port is absent.
- [ ] Implement enter and resume transactions:
```java
@Transactional ResumedAgentRun resumeConfirmationTx(ResumeConfirmationCommand c){
  AgentRun run=requireOwnedForUpdate(c.runId(),c.currentUserId());requireState(run,WAITING_CONFIRMATION);requireNotExpired(run,dbNow());
  PendingConfirmation pending=loadPendingConfirmation(c.runId(),c.replyId());if(!pending.decisionIds().equals(Set.copyOf(c.decisionIds())))throw invalidDecisions();
  long epoch=run.getOwnerEpoch()+1;run.resume(c.newOwnerInstanceId(),epoch,dbNow().plus(c.ownerLease()));runMapper.updateById(run);
  appendResumeAudit(c.runId(),"USER_CONFIRM_RESULT",c.replyId(),null);return resumed(run,epoch);
}
@Transactional boolean enterWaitingExternalTx(String runId,long expectedEpoch,WaitingCheckpoint cp,PendingExternalExecution p){
  AgentRun run=requireOwnedRunningForUpdate(runId,expectedEpoch);requireCheckpointIdentity(run,cp);requireFuture(p.expiresAt());
  run.enterExternal(p.toolCallId(),p.toolName(),p.expiresAt(),cp.pausedThroughSequence());runMapper.updateById(run);
  appendWaitingAudit(runId,"PLATFORM_REQUIRE_EXTERNAL_EXECUTION",p);return true;
}
```
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentWaitingStateIT" verify`; expect PASS; candidate is actionable only after state-save success, both resumes claim a new epoch/lease, persisted snapshot/session/deadline are byte-identical, and an unavailable snapshot is terminalized by the new owner as `RUN_CONFIG_UNAVAILABLE` through `terminateOwned` rather than `terminateSystem`.
- [ ] Commit with `git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentWaitingStatePort.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/DurableAgentWaitingStateService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/PendingConfirmation.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/PendingExternalExecution.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/WaitingCheckpoint.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/ResumeConfirmationCommand.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/ResumeExternalCommand.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/model/ResumedAgentRun.java ai-fusion-video/src/main/java/com/stonewu/fusion/mapper/ai/AgentRunMapper.java ai-fusion-video/src/main/java/com/stonewu/fusion/mapper/ai/AgentEventMapper.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentWaitingStateIT.java; git commit -m "feat(ai): persist recoverable waiting states"`.

### Task 10: Fence owners, reconcile leases, and cancel across nodes

**Files:** Create `service/ai/run/RunLeaseGuard.java`, `AgentRunReconciliationService.java`, `CancellationCoordinator.java`, `AgentRunMaintenanceScheduler.java`, `integration/AgentFencingCancellationIT.java`, `integration/AgentOwnedJournalTakeoverIT.java`; modify run mapper/repository, Redis signals, `RunShutdownCancellationPort.java`, `PlatformSubAgentRunService.java`, and `DefaultRunExecutionSupervisor.java`.

**Interfaces:** Produces `Mono<Void> assertLease(String runId,String ownerInstanceId,long ownerEpoch)`, guarded heartbeat, exact `Mono<AgentRunStatus> CancellationCoordinator.cancel(String runId,long currentUserId)`, and `CancellationCoordinator implements RunShutdownCancellationPort` for the supervisor's internal shutdown request. It interrupts through `OwnedExecutionRegistry`; the repository cancellation transaction locks the requested parent row, CASes that run, and CASes active descendants before releasing the row lock, serializing cancellation against Task 3 child admission. `CancellationCoordinator` is the only caller of `terminateSystem(...CANCELLATION_COORDINATOR)` and may request only `CANCEL_REQUESTED -> CANCELLED`; `AgentRunReconciliationService` is the only caller of `terminateSystem(...OWNER_RECONCILER)` and may request terminalization only after selecting an expired lease candidate. `MySqlRunTerminalCoordinator` independently locks and revalidates actor/status/database-time lease rules. Normal execution, resume failure, Provider failure, deadline, backpressure, and shutdown never call the system entry directly.

- [ ] Write tests for stale epoch side-effect rejection, Redis cancel loss observed by heartbeat, expired claim, cancel/complete race, parent cancellation serialized with concurrent child start, parent cancellation CASing child and grandchild runs, unrelated child trees untouched, and one terminal. Exercise both system actors directly: cancellation actor rejects every source state except `CANCEL_REQUESTED`; reconciler returns zero for non-expired `RUNNING` and non-expired `CANCEL_REQUESTED`, then produces the correct terminal only after DB time reaches `lease_until`. Add `AgentOwnedJournalTakeoverIT` with two real service contexts (`node-a`, `node-b`): node A owns epoch 1, the run enters WAITING, node B resumes and owns epoch 2, then node A's `appendOwned` and `terminateOwned` both return empty/affect zero rows while node B can append and create the sole terminal.
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentFencingCancellationIT" verify`; expect FAIL because current cancel uses conversation/local maps.
- [ ] Implement database-time guarded SQL and cancel CAS:
```sql
UPDATE afv_agent_run SET heartbeat_at=CURRENT_TIMESTAMP(3),lease_until=DATE_ADD(CURRENT_TIMESTAMP(3),INTERVAL ? MICROSECOND)
 WHERE run_id=? AND owner_instance_id=? AND owner_epoch=? AND status='RUNNING' AND lease_until>CURRENT_TIMESTAMP(3);
UPDATE afv_agent_run SET status='CANCEL_REQUESTED',cancel_requested_at=CURRENT_TIMESTAMP(3),cancel_next_attempt_at=CURRENT_TIMESTAMP(3)
 WHERE run_id=? AND user_id=? AND status IN('RUNNING','WAITING_CONFIRMATION','WAITING_EXTERNAL');
```
The authorized and internal cancellation transactions first `SELECT ... FOR UPDATE` the target run. While holding that row lock they perform the status CAS and `requestCancelActiveDescendants`; they return the root plus exact affected descendant owner tuples for post-commit Redis notification/interruption. No child scan or insertion occurs outside the shared parent-row serialization boundary.
- [ ] Implement heartbeat fallback so a lost Redis cancel signal is observed from MySQL within one heartbeat interval:
```java
public Mono<Void> heartbeat(String runId,String owner,long epoch,Duration lease){
  return journal(()->runRepository.renewOwnedLease(runId,owner,epoch,lease))
      .flatMap(updated->{
        if(updated==1) return Mono.empty();
        return journal(()->runRepository.requireByRunId(runId)).flatMap(current->{
          if(current.getStatus()==CANCEL_REQUESTED){
            return journal(()->runRepository.acknowledgeCancel(runId,owner,epoch))
                .then(executions.interruptOwned(runId,owner,epoch,CANCEL_REQUESTED)).then();
          }
          return executions.interruptOwned(runId,owner,epoch,OWNER_FENCED).then();
        });
      });
}
```
- [ ] Implement cancellation ordering so the parent and all active descendants are durably requested before local interruption is treated as complete:
```java
public Mono<AgentRunStatus> cancel(String runId,long currentUserId){
  return requestAuthorizedCancelTree(runId,currentUserId)
      .flatMap(tree->publishAndInterrupt(tree,CANCEL_REQUESTED)
          .then(tree.root().hasOwner()
              ? Mono.just(CANCEL_REQUESTED)
              : terminals.terminateSystem(RunTerminalRequest.cancelled(runId),
                    CANCELLATION_COORDINATOR).thenReturn(CANCELLED)));
}

@Override public Mono<Void> request(String runId){
  return requestInternalCancelTree(runId,"SHUTDOWN")
      .flatMap(tree->publishAndInterrupt(tree,SHUTDOWN).then());
}

public Mono<Void> reconcileExpiredOwner(String runId){
  return repository.lockAndClassifyExpired(runId)
      .flatMap(expired->expired.cancelRequested()
          ? terminals.terminateSystem(RunTerminalRequest.cancelled(runId),OWNER_RECONCILER).then()
          : terminals.terminateSystem(RunTerminalRequest.ownerLost(runId),OWNER_RECONCILER).then());
}
```
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentFencingCancellationIT,AgentOwnedJournalTakeoverIT" verify`; expect PASS: old owner journal/terminal updates are zero after node B takeover, lost Redis still cancels, child start cannot race past locked parent cancellation, parent cancellation propagates to descendants, non-expired system terminal attempts affect zero rows, and an expired non-cancelled owner becomes `FAILED/OWNER_LOST` without re-execution.
- [ ] Commit with `git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/RunLeaseGuard.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRunReconciliationService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/CancellationCoordinator.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRunMaintenanceScheduler.java ai-fusion-video/src/main/java/com/stonewu/fusion/repository/ai/AgentRunRepository.java ai-fusion-video/src/main/java/com/stonewu/fusion/mapper/ai/AgentRunMapper.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRunRedisSignalService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/RunShutdownCancellationPort.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/PlatformSubAgentRunService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/DefaultRunExecutionSupervisor.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentFencingCancellationIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentOwnedJournalTakeoverIT.java; git commit -m "feat(ai): fence owners and coordinate cancellation"`.

### Task 11: Switch replay to MySQL truth

**Files:** Create `service/ai/run/ReplayWakeGate.java`, `AgentRunReplayService.java`, `integration/AgentReplayLiveIT.java`; modify event repository and Redis signals.

**Interfaces:** Produces `Flux<CommittedAgentEvent> replayThenLive(String runId,long afterSequence)`.

- [ ] Write tests inserting events after subscribe, watermark, replay, and empty tail; also drop/reorder/duplicate Redis wake-ups.
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentReplayLiveIT" verify`; expect FAIL because current reconnect reads Redis Replay List and returns empty after terminal.
- [ ] Implement subscribe-before-watermark and DB tail:
```java
final class ReplayWakeGate {
  private final AtomicBoolean dirty = new AtomicBoolean();
  private final Sinks.Many<Long> wake = Sinks.many().multicast().directBestEffort();
  void markDirty(long sequenceHint) {
    dirty.set(true);
    wake.tryEmitNext(sequenceHint);
  }
  Flux<Long> wakeups() {
    return Flux.defer(() -> Flux.concat(
        dirty.getAndSet(false) ? Mono.just(0L) : Mono.empty(),
        wake.asFlux().doOnNext(ignored -> dirty.set(false))));
  }
}

return Flux.defer(() -> {
  AtomicLong cursor = new AtomicLong(after);
  ReplayWakeGate gate = new ReplayWakeGate();
  return Flux.using(
      () -> signals.listen(runId, gate::markDirty),
      listener -> latest(runId)
          .flatMapMany(watermark -> readRange(runId, cursor, watermark))
          .concatWith(Flux.merge(gate.wakeups(), Flux.interval(Duration.ofSeconds(2)))
              .concatMap(ignored -> readTail(runId, cursor)))
          .takeUntilOther(terminalDeliveredAndEmptyTail(runId, cursor)),
      RedisWakeListener::close);
});
```
- [ ] Run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentReplayLiveIT" verify`; expect PASS with ordered, duplicate-free delivery and close only after terminal plus empty tail.
- [ ] Commit with `git commit -m "feat(ai): replay events from MySQL truth"`.

### Task 12: Expose standard authorized SSE and cursor rules

**Files:** Create `service/ai/run/PipelineCursorParser.java`, `AgentRunQueryService.java`, `model/RunCursor.java`, status/running VOs, `test/.../controller/ai/AiPipelineSseControllerTests.java`; modify `controller/ai/AiPipelineController.java`, `controller/ai/vo/AiChatStreamRespVO.java`.

**Interfaces:** Controller returns `Flux<ServerSentEvent<AiChatStreamRespVO>>`; cursor priority is authorized target run, matching header, matching query, else zero.

- [ ] Write wire tests for `id:run-1:8`, `data:`, header/query conflict 400, header run mismatch 400, cross-user 404, and terminal replay.
- [ ] Run `.\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AiPipelineSseControllerTests" test`; expect FAIL because current controller emits data-only VO and lacks cursor authorization.
- [ ] Implement cursor parser and SSE mapping:
```java
RunCursor parse(String runId,Long query,String header){if(header==null)return new RunCursor(runId,query==null?0:nonNegative(query));
 int p=header.lastIndexOf(':');if(p<=0||!header.substring(0,p).equals(runId))throw invalidCursor();long h=parseNonNegative(header.substring(p+1));
 if(query!=null&&query.longValue()!=h)throw invalidCursor();return new RunCursor(runId,h);}
ServerSentEvent<AiChatStreamRespVO> toSse(CommittedAgentEvent e){return ServerSentEvent.<AiChatStreamRespVO>builder(e.projection()).id(e.runId()+":"+e.sequence()).event("pipeline-event").build();}
```
- [ ] Run `.\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AiPipelineSseControllerTests" test`; expect PASS and ownership applied to run/cancel/status/running/reconnect/confirm.
- [ ] Commit with `git commit -m "feat(api): expose cursor-correct pipeline SSE"`.

### Task 13: Prove the complete runtime with two real instances

**Files:** Create `service/ai/run/AgentRuntimeMetrics.java`, `test/.../run/AgentRuntimeMetricsTests.java`, `test/.../run/DurableRuntimeRequiredTestsContractTests.java`, `integration/support/AgentRuntimeContainers.java`, `integration/AgentIntegrationProfileSentinelIT.java`, `integration/AgentDurableRuntimeMultiInstanceIT.java`; modify `service/ai/run/MySqlAgentEventJournal.java`, `MySqlRunTerminalCoordinator.java`, `AgentEventOutboxPublisher.java`, `AgentMessageProjectionService.java`, `OwnedExecutionRegistry.java`, `DefaultRunExecutionSupervisor.java`, `PlatformSubAgentRunService.java`, `RunLeaseGuard.java`, `CancellationCoordinator.java`, `AgentRunReconciliationService.java`, `AgentRunReplayService.java`, `service/ai/agentscope/state/FailClosedAgentStateStore.java`, `AgentStatePreflight.java`, `service/ai/agentscope/kernel/HarnessLeaseCache.java`, `service/ai/agentscope/runtime/AgentRuntimeSchedulers.java`, and every exact Failsafe IT listed in the Task 1 profile to emit bounded-cardinality metrics and prohibit skipped execution.

**Interfaces:** Exposes Micrometer counters/timers/gauges for active/waiting/lost runs, Harness hit/miss/eviction/active lease/capacity rejection, StateStore latency/failure/bulkhead, event persist/sequence/backpressure, outbox backlog/retry, Provider terminal/close, tool scheduler violations, replay/dedup/terminal recovery; tags never include runId, conversationId, userId, toolCallId, model prompt, or secrets. Uses MySQL `8.4.6`, Redis `7.4.5-alpine`, and two coordinators with `node-a/node-b`; no Mockito proves distributed properties.

- [ ] Write `AgentRuntimeMetricsTests` with `SimpleMeterRegistry` to assert every required metric is registered, increments at the owning boundary, and rejects high-cardinality identity tags; then write one end-to-end test: start on A, journal/projection, WAITING resume on B, stale A fenced, cancel signal dropped, heartbeat observes DB, outbox takeover, replay terminal from B. Add a Surefire contract test that makes removal or renaming of a required IT fail before Failsafe:
```java
@ParameterizedTest
@ValueSource(strings={
  "AgentPersistenceMigrationIT","AgentMessageAllocatorIT","AgentRunStartIT",
  "AgentJournalTerminalIT","AgentModelCallUsageIT","AgentOutboxMultiInstanceIT",
  "AgentProjectionRecoveryIT","PlatformSubAgentRunServiceIT","AgentWaitingStateIT",
  "AgentFencingCancellationIT","AgentOwnedJournalTakeoverIT","AgentReplayLiveIT",
  "AgentIntegrationProfileSentinelIT",
  "AgentDurableRuntimeMultiInstanceIT"})
void requiredIntegrationTestExistsAndIsEnabled(String simpleName){
  Class<?> type=assertDoesNotThrow(()->Class.forName("com.stonewu.fusion.integration."+simpleName));
  Set<String> conditional=Set.of("Disabled","DisabledIf","EnabledIf",
      "DisabledIfEnvironmentVariable","EnabledIfEnvironmentVariable",
      "DisabledIfSystemProperty","EnabledIfSystemProperty","DisabledOnOs","EnabledOnOs",
      "DisabledOnJre","EnabledOnJre","DisabledForJreRange","EnabledForJreRange");
  assertThat(Arrays.stream(type.getAnnotations())
      .map(a->a.annotationType().getSimpleName())).doesNotContainAnyElementsOf(conditional);
  List<Method> tests=Arrays.stream(type.getDeclaredMethods())
      .filter(m->AnnotationSupport.isAnnotated(m,Test.class)
          || AnnotationSupport.isAnnotated(m,TestTemplate.class)).toList();
  assertThat(tests).isNotEmpty();
  assertThat(tests).allSatisfy(method->assertThat(Arrays.stream(method.getAnnotations())
      .map(a->a.annotationType().getSimpleName())).doesNotContainAnyElementsOf(conditional));
  String source=Files.readString(Path.of("src/test/java/com/stonewu/fusion/integration/"
      +simpleName+".java"));
  assertThat(source).doesNotContain("Assumptions.","assumeTrue(","assumeFalse(",
      "TestAbortedException","disabledWithoutDocker = true","disabledWithoutDocker=true");
}
```
- [ ] Add a real profile sentinel; it contains no condition or assumption and Docker absence is a hard failure:
```java
@Testcontainers(disabledWithoutDocker = false)
class AgentIntegrationProfileSentinelIT {
  @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6");
  @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine")
      .withExposedPorts(6379);

  @Test void startsRealMysqlAndRedis() throws Exception {
    assertThat(MYSQL.isRunning()).isTrue();
    assertThat(REDIS.isRunning()).isTrue();
    try (Connection c=DriverManager.getConnection(MYSQL.getJdbcUrl(),MYSQL.getUsername(),MYSQL.getPassword());
         Statement s=c.createStatement(); ResultSet rs=s.executeQuery("SELECT VERSION()")) {
      assertThat(rs.next()).isTrue();
      assertThat(rs.getString(1)).startsWith("8.");
    }
    try (Jedis jedis=new Jedis(REDIS.getHost(),REDIS.getMappedPort(6379))) {
      assertThat(jedis.ping()).isEqualTo("PONG");
    }
  }
}
```
Every other required IT also uses `@Testcontainers(disabledWithoutDocker = false)` where applicable and contains no `@Disabled`, JUnit conditional-enable/disable annotation, `Assumptions`, `assumeTrue/assumeFalse`, or `TestAbortedException` at class or method scope.
- [ ] Run `.\mvnw.cmd "-Dsurefire.failIfNoSpecifiedTests=true" "-Dtest=AgentRuntimeMetricsTests,DurableRuntimeRequiredTestsContractTests" test`; expect FAIL because the metric facade/sentinel contract is absent. Then run `.\mvnw.cmd -Pagentscope-integration "-Dfailsafe.failIfNoSpecifiedTests=true" "-Dit.test=AgentIntegrationProfileSentinelIT,AgentDurableRuntimeMultiInstanceIT" verify`; expect FAIL on any remaining real configuration/timing gap, FAIL if either named IT is missing, and fail rather than skip when Docker is unavailable.
- [ ] Use event-driven assertions without sleeps:
```java
await().atMost(Duration.ofSeconds(10)).pollInterval(Duration.ofMillis(50)).untilAsserted(()->{
 assertThat(nodeB.status("run-1")).isEqualTo(CANCELLED);assertThat(nodeB.terminalCount("run-1")).isEqualTo(1);
 assertThat(nodeB.unpublishedRequired("run-1")).isZero();assertThat(nodeB.projectedThrough("run-1")).isEqualTo(nodeB.terminalSequence("run-1"));
});
```
- [ ] Wire the metric facade through explicit methods rather than ad-hoc meter names:
```java
public void eventPersisted(Duration latency) {
  eventPersistTimer.record(latency);
  eventPersistedCounter.increment();
}
public void terminal(AgentRunStatus status) {
  terminalCounters.get(status).increment();
}
public void outboxBacklog(long count) {
  outboxBacklog.set(count);
}
```
- [ ] Run the single fail-fast phase gate from `D:\develop\my\ai-fusion-video\ai-fusion-video`:
```powershell
.\mvnw.cmd clean -Pagentscope-integration -DskipTests=false -DskipITs=false verify
if ($LASTEXITCODE -ne 0) { throw 'durable runtime Maven phase gate failed' }
$expected = @(
  'AgentPersistenceMigrationIT','AgentMessageAllocatorIT','AgentRunStartIT',
  'AgentJournalTerminalIT','AgentModelCallUsageIT','AgentOutboxMultiInstanceIT',
  'AgentProjectionRecoveryIT','PlatformSubAgentRunServiceIT','AgentWaitingStateIT',
  'AgentFencingCancellationIT','AgentOwnedJournalTakeoverIT','AgentReplayLiveIT',
  'AgentIntegrationProfileSentinelIT','AgentDurableRuntimeMultiInstanceIT')
$reports = Get-ChildItem -LiteralPath 'target/failsafe-reports' -Filter 'TEST-*.xml' -File
$actual = @($reports | ForEach-Object { ([xml](Get-Content -Raw -LiteralPath $_.FullName)).testsuite.name.Split('.')[-1] } | Sort-Object -Unique)
$difference = @(Compare-Object ($expected | Sort-Object) $actual)
if ($difference.Count -ne 0) { $difference; throw 'Failsafe report set differs from the exact required IT list' }
$tests = 0; $skipped = 0
foreach ($report in $reports) {
  $suite = ([xml](Get-Content -Raw -LiteralPath $report.FullName)).testsuite
  $tests += [int]$suite.tests
  $skipped += [int]$suite.skipped
}
if ($tests -le 0) { throw 'Failsafe executed zero integration tests' }
if ($skipped -ne 0) { throw "Failsafe skipped $skipped integration tests" }
"Failsafe required reports=$($reports.Count), tests=$tests, skipped=$skipped"
```
Expected: one clean Maven reactor ends with `BUILD SUCCESS`; Surefire unit tests, package, the exact fourteen Failsafe IT classes, and Failsafe `verify` all run. The XML audit prints fourteen required reports, positive test count, and `skipped=0`. A missing IT/report, extra wildcard-discovered IT, unavailable Docker daemon, conditional disable, assumption abort, skipped test, compilation failure, unit failure, or integration failure produces a non-zero gate.
- [ ] Commit with every production and test file owned by Task 13 staged explicitly:
```powershell
git add ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRuntimeMetrics.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/MySqlAgentEventJournal.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/MySqlRunTerminalCoordinator.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentEventOutboxPublisher.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentMessageProjectionService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/OwnedExecutionRegistry.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/DefaultRunExecutionSupervisor.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/PlatformSubAgentRunService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/RunLeaseGuard.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/CancellationCoordinator.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRunReconciliationService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/run/AgentRunReplayService.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/FailClosedAgentStateStore.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/state/AgentStatePreflight.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/kernel/HarnessLeaseCache.java ai-fusion-video/src/main/java/com/stonewu/fusion/service/ai/agentscope/runtime/AgentRuntimeSchedulers.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/AgentRuntimeMetricsTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/service/ai/run/DurableRuntimeRequiredTestsContractTests.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/support/AgentRuntimeContainers.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentPersistenceMigrationIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentMessageAllocatorIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentRunStartIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentJournalTerminalIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentModelCallUsageIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentOutboxMultiInstanceIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentProjectionRecoveryIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/PlatformSubAgentRunServiceIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentWaitingStateIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentFencingCancellationIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentOwnedJournalTakeoverIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentReplayLiveIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentIntegrationProfileSentinelIT.java ai-fusion-video/src/test/java/com/stonewu/fusion/integration/AgentDurableRuntimeMultiInstanceIT.java
git commit -m "test(ai): verify durable runtime across instances"
```

## Rollback Gate

Stop Pipeline writes, terminalize/cancel active V2 runs, stop maintenance workers, drain or retain required outbox rows, then start V1; never reverse Flyway. Before re-enabling V2, stop writes and reconcile counters including deleted messages:

```sql
UPDATE afv_agent_conversation c LEFT JOIN
 (SELECT conversation_id,COALESCE(MAX(message_order),0)+1 n FROM afv_agent_message GROUP BY conversation_id) m
 ON m.conversation_id=c.conversation_id
SET c.next_message_order=GREATEST(c.next_message_order,COALESCE(m.n,1));
```
