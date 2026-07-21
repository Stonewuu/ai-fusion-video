package com.stonewu.fusion.repository.ai;

import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.SystemTerminalActor;

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
}
