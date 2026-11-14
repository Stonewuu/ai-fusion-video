package com.stonewu.fusion.service.execution;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.job.GenerationAttemptRepository;
import com.stonewu.fusion.service.generation.job.GenerationExecutionInput;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;

/** 仅在异常记录的所属用户校验通过后读取，不公开模型鉴权及供应商原始响应。 */
@Component
@RequiredArgsConstructor
public class ExecutionRecordGenerationDetails {
    private final GenerationJobRepository jobs;
    private final GenerationAttemptRepository attempts;
    private final CanvasJson json;

    public View get(String jobId) {
        var job = jobs.get(jobId);
        var submission = jobs.submission(job.submissionId());
        return new View(submission.kind(), submission.model().getName(), submission.model().getCode(),
                GenerationExecutionInput.from(job.input(), job.itemIds().size(), json), job.phase(), job.pollAttempts(),
                job.result() != null && job.result().state() == RemoteGenerationResult.State.SUCCEEDED,
                attempts.forJob(jobId).stream().map(attempt -> new Attempt(attempt.id(), attempt.number(),
                        instant(attempt.submittedAt()), instant(attempt.acknowledgedAt()))).toList());
    }

    private static Instant instant(LocalDateTime value) {
        return value == null ? null : value.toInstant(ZoneOffset.UTC);
    }

    public record View(GenerationKind kind, String modelName, String modelCode, GenerationExecutionInput input,
                       String phase, int pollAttempts, boolean resultReceived, List<Attempt> attempts) {
    }

    public record Attempt(String id, int number, Instant submittedAt, Instant acknowledgedAt) {
    }
}
