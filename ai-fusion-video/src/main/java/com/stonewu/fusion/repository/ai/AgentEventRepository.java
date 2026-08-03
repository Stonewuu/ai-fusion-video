package com.stonewu.fusion.repository.ai;

import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.SystemTerminalActor;

import java.util.List;
import java.util.Optional;

/** Synchronous short-transaction port implemented by the durable MySQL adapter. */
public interface AgentEventRepository {

    Optional<CommittedAgentEvent> appendOwnedTx(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            AgentEventEnvelope event);

    Optional<CommittedAgentEvent> terminateOwnedTx(
            RunTerminalRequest request,
            String ownerInstanceId,
            long ownerEpoch);

    Optional<CommittedAgentEvent> terminateSystemTx(
            RunTerminalRequest request,
            SystemTerminalActor actor);

    ReplaySnapshot loadReplaySnapshot(String runId);

    List<CommittedAgentEvent> loadReplayPage(
            String runId,
            long afterSequence,
            long throughSequence,
            int limit);

    /** A point-in-time view used to bound one replay drain. */
    record ReplaySnapshot(long latestSequence, Long terminalSequence) {

        public ReplaySnapshot {
            if (latestSequence < 0) {
                throw new IllegalArgumentException("latestSequence must not be negative");
            }
            if (terminalSequence != null
                    && (terminalSequence <= 0 || terminalSequence != latestSequence)) {
                throw new IllegalArgumentException(
                        "terminalSequence must identify the final committed event");
            }
        }
    }
}
