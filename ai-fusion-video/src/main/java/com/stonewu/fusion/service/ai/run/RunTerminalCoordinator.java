package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.SystemTerminalActor;
import reactor.core.publisher.Mono;

import java.util.Optional;

/** Reactive boundary for owner-fenced and allow-listed system terminal CAS operations. */
public interface RunTerminalCoordinator {

    Mono<Optional<CommittedAgentEvent>> terminateOwned(
            RunTerminalRequest request,
            String ownerInstanceId,
            long ownerEpoch);

    Mono<Optional<CommittedAgentEvent>> terminateSystem(
            RunTerminalRequest request,
            SystemTerminalActor actor);
}
