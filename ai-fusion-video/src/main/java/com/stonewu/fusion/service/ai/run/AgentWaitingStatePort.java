package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.run.model.PendingConfirmation;
import com.stonewu.fusion.service.ai.run.model.PendingExternalExecution;
import com.stonewu.fusion.service.ai.run.model.ResumeConfirmationCommand;
import com.stonewu.fusion.service.ai.run.model.ResumeExternalCommand;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import com.stonewu.fusion.service.ai.run.model.WaitingCheckpoint;
import reactor.core.publisher.Mono;

public interface AgentWaitingStatePort {

    Mono<Void> recordConfirmationCandidate(String runId, PendingConfirmation candidate);

    Mono<Boolean> enterWaitingConfirmation(
            String runId, long expectedOwnerEpoch, WaitingCheckpoint checkpoint);

    Mono<Boolean> enterWaitingExternal(
            String runId,
            long expectedOwnerEpoch,
            WaitingCheckpoint checkpoint,
            PendingExternalExecution pending);

    Mono<PendingConfirmation> getPendingConfirmationAuthorized(
            String runId, long currentUserId, String replyId);

    Mono<PendingExternalExecution> getPendingExternalAuthorized(
            String runId, long currentUserId, String toolCallId);

    Mono<ResumedAgentRun> resumeConfirmation(ResumeConfirmationCommand command);

    Mono<ResumedAgentRun> resumeExternal(ResumeExternalCommand command);
}
