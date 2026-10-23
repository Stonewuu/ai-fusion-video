package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.media.GenerationFrameService;
import com.stonewu.fusion.service.media.GenerationMediaService;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
@RequiredArgsConstructor
public class GenerationResultIntake {
    private final GenerationProviderInputs inputs;
    private final GenerationMediaService media;
    private final GenerationFrameService frames;
    private final GenerationJobRepository jobs;

    public List<RemoteGenerationResult.Output> receive(RemoteGenerationJob job, GenerationSubmission submission,
                                                       ApiConfig api) {
        var outputs = new ArrayList<>(job.result().outputs());
        if (outputs.size() > job.itemIds().size()) {
            throw new GenerationJobException(502, "GENERATION_OUTPUT_COUNT_MISMATCH",
                    "供应商输出超过已确认的数量，原响应已保留，没有自动取第一项");
        }
        for (int index = 0; index < outputs.size(); index++) {
            receiveOne(job, submission, api, outputs, index);
        }
        return List.copyOf(outputs);
    }

    private void receiveOne(RemoteGenerationJob job, GenerationSubmission submission, ApiConfig api,
                            List<RemoteGenerationResult.Output> outputs, int index) {
        var output = outputs.get(index);
        long itemId = job.itemIds().get(index);
        String prefix = "item-" + itemId;
        if (submission.kind() == GenerationKind.IMAGE) {
            var task = inputs.image(job, submission, api, false);
            String image = media.image(task, prefix + "-image", output.media(), api);
            checkpoint(job, outputs, index, new RemoteGenerationResult.Output(image, output.cover(), null, null, null));
            String thumbnail = output.cover() == null ? null : media.image(task, prefix + "-thumbnail", output.cover(),
                    api);
            checkpoint(job, outputs, index, new RemoteGenerationResult.Output(image, thumbnail, null, null, null));
            return;
        }
        var task = inputs.video(job, submission, api, false);
        String video = media.video(task, prefix + "-video", output.media(), api, MediaKind.VIDEO);
        checkpoint(job, outputs, index,
                new RemoteGenerationResult.Output(video, output.cover(), output.firstFrame(), output.lastFrame(),
                        output.durationSeconds()));
        String first = output.firstFrame() == null ? null : media.video(task, prefix + "-firstFrame",
                output.firstFrame(), api, MediaKind.IMAGE);
        checkpoint(job, outputs, index,
                new RemoteGenerationResult.Output(video, output.cover(), first, output.lastFrame(),
                        output.durationSeconds()));
        String last = output.lastFrame() == null ? null : media.video(task, prefix + "-lastFrame", output.lastFrame(),
                api, MediaKind.IMAGE);
        checkpoint(job, outputs, index,
                new RemoteGenerationResult.Output(video, output.cover(), first, last, output.durationSeconds()));
        String cover = output.cover() == null ? null : media.video(task, prefix + "-cover", output.cover(), api,
                MediaKind.IMAGE);
        checkpoint(job, outputs, index,
                new RemoteGenerationResult.Output(video, cover, first, last, output.durationSeconds()));
        var extracted = frames.extract(task,
                VideoItem.builder().id(itemId).taskId(submission.taskId()).videoUrl(video).build(), first == null,
                last == null);
        if (first == null) {
            first = extracted.firstFrameUrl();
        }
        if (last == null) {
            last = extracted.lastFrameUrl();
        }
        checkpoint(job, outputs, index,
                new RemoteGenerationResult.Output(video, cover == null ? first : cover, first, last,
                        output.durationSeconds()));
    }

    private void checkpoint(RemoteGenerationJob job, List<RemoteGenerationResult.Output> outputs, int index,
                            RemoteGenerationResult.Output output) {
        outputs.set(index, output);
        jobs.remoteResult(job.claim(), RemoteGenerationResult.success(outputs));
    }
}
