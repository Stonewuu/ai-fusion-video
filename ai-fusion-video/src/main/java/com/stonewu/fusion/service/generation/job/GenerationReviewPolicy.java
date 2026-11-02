package com.stonewu.fusion.service.generation.job;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public class GenerationReviewPolicy {
    public List<GenerationJobCommands.Action> allowed(RemoteGenerationJob job,
                                                      List<GenerationAttemptRepository.Summary> attempts) {
        if (job.state().equals("FAILED") && !job.cancelRequested() && job.result() != null
                && job.result().state() == RemoteGenerationResult.State.SUCCEEDED) {
            return List.of(GenerationJobCommands.Action.RETRY_INGEST);
        }
        return List.of();
    }

    public List<GenerationAttemptRepository.Summary> lateReceipts(RemoteGenerationJob job,
                                                                  List<GenerationAttemptRepository.Summary> attempts) {
        return attempts.stream().filter(attempt -> !attempt.id().equals(job.activeAttemptId()) && attempt.acknowledged()
                && "CONFIRM_NOT_ACCEPTED_AND_RETRY".equals(attempt.resolution())).toList();
    }
}
