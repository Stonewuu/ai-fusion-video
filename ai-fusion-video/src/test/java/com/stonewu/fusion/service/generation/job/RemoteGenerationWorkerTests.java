package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategyRouter;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategyRouter;
import com.stonewu.fusion.service.media.ManagedStorageConfiguration;
import com.stonewu.fusion.service.media.model.MediaException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** 保护配置错误立即进入终态，同时保留真正临时故障的保存重试。 */
class RemoteGenerationWorkerTests {
    private final GenerationJobRepository jobs = mock(GenerationJobRepository.class);
    private final GenerationAttemptRepository attempts = mock(GenerationAttemptRepository.class);
    private final GenerationJobLifecycle lifecycle = mock(GenerationJobLifecycle.class);
    private final GenerationExecutionIdentity identity = mock(GenerationExecutionIdentity.class);
    private final GenerationModelSnapshot models = mock(GenerationModelSnapshot.class);
    private final GenerationResultIntake intake = mock(GenerationResultIntake.class);
    private final RemoteGenerationJob job = mock(RemoteGenerationJob.class);
    private final GenerationSubmission submission = mock(GenerationSubmission.class);
    private final ApiConfig api = ApiConfig.builder().id(1L).build();
    private final RemoteGenerationJob.Claim claim = new RemoteGenerationJob.Claim("job", "worker", 1);
    private final RemoteGenerationWorker worker = new RemoteGenerationWorker(jobs, attempts, lifecycle, identity,
            models, mock(GenerationProviderInputs.class), intake, mock(ImageGenerationStrategyRouter.class),
            mock(VideoGenerationStrategyRouter.class), mock(ManagedStorageConfiguration.class));

    @BeforeEach
    void prepare() {
        when(jobs.claim(anyString())).thenReturn(Optional.of(job));
        when(job.id()).thenReturn("job");
        when(job.claim()).thenReturn(claim);
        when(job.phase()).thenReturn("INGESTING");
        when(job.deadline()).thenReturn(LocalDateTime.of(2030, 1, 1, 0, 0));
        when(jobs.get("job")).thenReturn(job);
        when(jobs.now()).thenReturn(LocalDateTime.of(2026, 11, 14, 0, 0));
        when(lifecycle.start(claim)).thenReturn(submission);
        when(identity.permitted(submission)).thenReturn(true);
        when(models.forExecution(submission)).thenReturn(api);
        when(job.result()).thenReturn(RemoteGenerationResult.success(
                List.of(RemoteGenerationResult.Output.image("data:image/png;base64,saved-result"))));
    }

    @AfterEach
    void close() {
        worker.stop();
    }

    @Test
    void invalidStorageEndsImmediatelyAndPreservesTheReceivedResult() {
        when(intake.receive(job, submission, api))
                .thenThrow(MediaException.storageConfiguration("默认存储未启用"));
        assertThat(worker.runOne()).isTrue();
        verify(lifecycle).fail(claim, "FAILED", "MEDIA_STORAGE_CONFIGURATION_INVALID", "默认存储未启用");
        verify(jobs, never()).waitForNextPoll(any(), anyInt(), anyString(), any(), any());
        verifyNoInteractions(attempts);
        assertThat(job.result().outputs()).hasSize(1);
    }

    @ParameterizedTest
    @ValueSource(strings = {"SUBMITTING", "POLLING"})
    void changedProviderDoesNotRequeueEvenWhileReconcilingSubmission(String phase) {
        when(job.phase()).thenReturn(phase);
        when(job.result()).thenReturn(null);
        when(models.forExecution(submission)).thenThrow(new GenerationJobException(409,
                "GENERATION_PROVIDER_CHANGED", "供应商连接配置已变化"));
        worker.runOne();
        verify(lifecycle).fail(claim, "UNKNOWN_SUBMISSION", "GENERATION_PROVIDER_CHANGED", "供应商连接配置已变化");
        verify(jobs, never()).waitForNextPoll(any(), anyInt(), anyString(), any(), any());
        verifyNoInteractions(attempts, intake);
    }

    @Test
    void cancelledPreparationRemainsCancelledWhenConfigurationValidationFails() {
        var cancelled = new AtomicBoolean();
        when(job.phase()).thenReturn("PREPARING");
        when(job.result()).thenReturn(null);
        when(job.cancelRequested()).thenAnswer(ignored -> cancelled.get());
        when(models.forExecution(submission)).thenAnswer(ignored -> {
            cancelled.set(true);
            throw new GenerationJobException(409, "GENERATION_PROVIDER_CHANGED", "供应商连接配置已变化");
        });

        worker.runOne();

        verify(lifecycle).fail(claim, "CANCELLED", "GENERATION_PROVIDER_CHANGED", "供应商连接配置已变化");
        verify(jobs, never()).waitForNextPoll(any(), anyInt(), anyString(), any(), any());
        verifyNoInteractions(attempts, intake);
    }

    @Test
    void temporaryMediaTransferFailureRetriesTheSavedResponseWithoutResubmission() {
        when(intake.receive(job, submission, api))
                .thenThrow(new MediaException(502, "GENERATION_MEDIA_FAILED", "图片下载超时"));
        worker.runOne();
        verify(jobs).waitForNextPoll(claim, 2, "INGESTING", "GENERATION_MEDIA_FAILED", "图片下载超时");
        verify(lifecycle, never()).fail(any(), anyString(), anyString(), anyString());
        verifyNoInteractions(attempts);
    }
}
