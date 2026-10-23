package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.StoryboardMutationRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.STORYBOARD_EPISODE;

@Component
@RequiredArgsConstructor
public class ComposeWritebackHandler implements ResourceCommandHandler {
    private final StoryboardCommandSupport support;
    private final StoryboardMutationRepository writes;
    private final StoryboardVersionView versions;
    private final MediaJobRepository jobs;
    private final MediaJobFence fence;

    @Override
    public CanvasValueType resourceType() {
        return STORYBOARD_EPISODE;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.COMPOSE_WRITEBACK;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        return support.authorize(STORYBOARD_EPISODE, ((ComposeWritebackCommand) input).episodeId(), context, false);
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ComposeWritebackCommand) input;
        MediaJob job = jobs.get(command.jobId());
        if (job.targetId() != command.episodeId() || !job.context().equals(context)) {
            throw MediaException.invalid("合成任务归属不一致");
        }
        boolean starting = command.stage() == ComposeWritebackCommand.Stage.START;
        if (starting) {
            if (!job.status().equals("QUEUED") || job.epoch() != 0) {
                throw MediaJobFence.lost();
            }
        } else {
            if (command.claim() == null || !command.claim().equals(job.claim())) {
                throw MediaJobFence.lost();
            }
            fence.require(command.claim(), context, command.stage() == ComposeWritebackCommand.Stage.END);
        }
        var change = writes.lock(STORYBOARD_EPISODE, command.episodeId());
        var current = change.data(change.target);
        if (!starting && !command.jobId().equals(current.path("composeJobId").asText())) {
            throw MediaJobFence.lost();
        }
        Map<String, Object> fields = new LinkedHashMap<>();
        if (starting) {
            ResourceMutations.requireVersion(change.target, job.targetVersion() - 1);
            fields.put("composeJobId", job.id());
            fields.put("composeStatus", 1);
            fields.put("composeErrorMsg", null);
        } else if (command.stage() == ComposeWritebackCommand.Stage.COMPLETE) {
            ResourceMutations.requireVersion(change.target, job.targetVersion());
            String video = job.checkpoint().path("video").asText();
            MediaReference.id(video);
            fields.put("composedVideoUrl", video);
            fields.put("composedAt", LocalDateTime.now());
            fields.put("composeStatus", 2);
            fields.put("composeErrorMsg", null);
            fields.put("bgmAudioUrl", optional(job, "music"));
            fields.put("bgmSfxAudioUrl", optional(job, "sfx"));
            fields.put("bgmErrorMsg", null);
            fields.put("bgmTaskId",
                    jobs.submission(job.id(), "music").map(MediaJobRepository.Submission::providerId).orElse(null));
        } else {
            // 结束自己的执行状态不会覆盖镜头、文案及已有成片；完成产物仍只按提交版本写回。
            fields.put("composeStatus", 3);
            fields.put("composeErrorMsg", job.error());
        }
        change.patch(change.target, fields);
        change.finish();
        return change.result(
                versions.attach(STORYBOARD_EPISODE, change.current(STORYBOARD_EPISODE, command.episodeId())));
    }

    private String optional(MediaJob job, String field) {
        String value = job.checkpoint().path(field).asText(null);
        if (value != null) {
            MediaReference.id(value);
        }
        return value;
    }
}
