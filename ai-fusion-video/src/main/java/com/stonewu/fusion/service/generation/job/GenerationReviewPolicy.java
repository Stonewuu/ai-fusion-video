package com.stonewu.fusion.service.generation.job;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public class GenerationReviewPolicy {
    public List<GenerationJobCommands.Action> allowed(RemoteGenerationJob job,
                                                      List<GenerationAttemptRepository.Summary> attempts) {
        var actions = new ArrayList<GenerationJobCommands.Action>();
        if (!job.requiresReview() && !job.state().equals("FAILED")) {
            return actions;
        }
        if (job.state().equals("FAILED") && job.result() != null
                && job.result().state() == RemoteGenerationResult.State.FAILED) {
            return actions;
        }
        var active = attempts.stream().filter(attempt -> attempt.id().equals(job.activeAttemptId())).findFirst()
                .orElse(null);
        if (active != null && (active.acknowledged() || active.providerTaskId() != null)) {
            actions.add(GenerationJobCommands.Action.RECHECK);
        }
        if (!job.cancelRequested() && job.result() != null && job.result()
                .state() == RemoteGenerationResult.State.SUCCEEDED) {
            actions.add(GenerationJobCommands.Action.RETRY_INGEST);
        }
        if (job.requiresReview()) {
            if (!job.cancelRequested() && active != null && active.submitted() && !active.acknowledged()) {
                actions.add(GenerationJobCommands.Action.CONFIRM_NOT_ACCEPTED_AND_RETRY);
            }
            if ((active == null || !active.submitted()) && lateReceipts(job, attempts).size() == 1) {
                actions.add(GenerationJobCommands.Action.RESTORE_ACCEPTED_RECEIPT);
            }
            actions.add(GenerationJobCommands.Action.CONFIRM_PROVIDER_FINISHED);
        }
        return actions;
    }

    public List<GenerationAttemptRepository.Summary> lateReceipts(RemoteGenerationJob job,
                                                                  List<GenerationAttemptRepository.Summary> attempts) {
        return attempts.stream().filter(attempt -> !attempt.id().equals(job.activeAttemptId()) && attempt.acknowledged()
                && "CONFIRM_NOT_ACCEPTED_AND_RETRY".equals(attempt.resolution())).toList();
    }
}
