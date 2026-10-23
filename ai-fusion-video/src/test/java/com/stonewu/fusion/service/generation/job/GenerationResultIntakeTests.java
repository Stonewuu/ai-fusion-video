package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.media.GenerationFrameService;
import com.stonewu.fusion.service.media.GenerationMediaService;
import com.stonewu.fusion.service.media.model.MediaKind;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class GenerationResultIntakeTests {
    private static final String VIDEO = "media://00000000-0000-0000-0000-000000000001";
    private static final String FIRST = "media://00000000-0000-0000-0000-000000000002";
    private static final String LAST = "media://00000000-0000-0000-0000-000000000003";

    @Test void missingFramesAreDerivedAndTheManagedVideoIsCheckpointedBeforeFrameProcessing() {
        var f = fixture();
        var saved = new AtomicReference<RemoteGenerationResult>();
        doAnswer(call -> { saved.set(call.getArgument(1)); return null; }).when(f.jobs).remoteResult(any(), any());
        when(f.frames.extract(eq(f.task), any(), eq(true), eq(true))).thenAnswer(call -> {
            assertThat(saved.get().outputs().getFirst().media()).isEqualTo(VIDEO);
            assertThat(saved.get().outputs().getFirst().firstFrame()).isNull();
            return new GenerationFrameService.Frames(FIRST, LAST);
        });
        var outputs = f.intake.receive(f.job, f.submission, null);
        assertThat(outputs).containsExactly(new RemoteGenerationResult.Output(VIDEO, FIRST, FIRST, LAST, 4));
        assertThat(saved.get()).isEqualTo(RemoteGenerationResult.success(outputs));
    }

    @Test void frameFailurePreservesDownloadedVideoCheckpointForRecovery() {
        var f = fixture();
        when(f.frames.extract(eq(f.task), any(), eq(true), eq(true))).thenThrow(new IllegalStateException("FFmpeg 暂不可用"));
        assertThatThrownBy(() -> f.intake.receive(f.job, f.submission, null)).hasMessageContaining("FFmpeg");
        var checkpoint = ArgumentCaptor.forClass(RemoteGenerationResult.class);
        verify(f.jobs, atLeastOnce()).remoteResult(eq(f.job.claim()), checkpoint.capture());
        assertThat(checkpoint.getValue().outputs().getFirst().media()).isEqualTo(VIDEO);
        assertThat(checkpoint.getValue().outputs().getFirst().firstFrame()).isNull();
    }

    private Fixture fixture() {
        var job = mock(RemoteGenerationJob.class);
        var submission = mock(GenerationSubmission.class);
        var inputs = mock(GenerationProviderInputs.class);
        var media = mock(GenerationMediaService.class);
        var frames = mock(GenerationFrameService.class);
        var jobs = mock(GenerationJobRepository.class);
        var task = VideoTask.builder().id(201L).count(1).build();
        when(job.claim()).thenReturn(new RemoteGenerationJob.Claim("job", "worker", 1));
        when(job.itemIds()).thenReturn(List.of(301L));
        when(job.result()).thenReturn(RemoteGenerationResult.success(List.of(RemoteGenerationResult.Output.video("https://example.com/video.mp4", null, 4))));
        when(submission.kind()).thenReturn(GenerationKind.VIDEO);
        when(submission.taskId()).thenReturn(201L);
        when(inputs.video(job, submission, null, false)).thenReturn(task);
        when(media.video(eq(task), eq("item-301-video"), anyString(), isNull(), eq(MediaKind.VIDEO))).thenReturn(VIDEO);
        return new Fixture(job, submission, jobs, task, frames, new GenerationResultIntake(inputs, media, frames, jobs));
    }
    private record Fixture(RemoteGenerationJob job, GenerationSubmission submission, GenerationJobRepository jobs,
                           VideoTask task, GenerationFrameService frames, GenerationResultIntake intake) { }
}
