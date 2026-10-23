package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.StoryboardMutationRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.job.EpisodeComposeGuard;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.STORYBOARD_EPISODE;

@Component
@RequiredArgsConstructor
public class AdoptEpisodeMediaHandler implements ResourceCommandHandler {
    private final StoryboardCommandSupport support;
    private final StoryboardMutationRepository writes;
    private final StoryboardVersionView versions;
    private final EpisodeComposeGuard activeJobs;

    @Override
    public CanvasValueType resourceType() {
        return STORYBOARD_EPISODE;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.ADOPT_EPISODE_MEDIA;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        return support.authorize(STORYBOARD_EPISODE, ((AdoptEpisodeMediaCommand) input).episodeId(), context, false);
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (AdoptEpisodeMediaCommand) input;
        if (!Set.of("composedVideoUrl", "bgmAudioUrl", "bgmSfxAudioUrl").contains(command.field())) {
            throw MediaException.invalid("不支持的分集媒体字段");
        }
        var change = writes.lock(STORYBOARD_EPISODE, command.episodeId());
        ResourceMutations.requireVersion(change.target, command.expectedVersion());
        activeJobs.requireIdle(command.episodeId(), change.data(change.target));
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put(command.field(), MediaReference.of(command.mediaId()));
        if (command.field().equals("composedVideoUrl")) {
            fields.put("composeJobId", null);
            fields.put("composeStatus", 2);
            fields.put("composeErrorMsg", null);
            fields.put("composedAt", LocalDateTime.now());
        } else {
            fields.put("bgmTaskId", null);
            fields.put("bgmErrorMsg", null);
        }
        change.patch(change.target, fields);
        change.finish();
        return change.result(
                versions.attach(STORYBOARD_EPISODE, change.current(STORYBOARD_EPISODE, command.episodeId())));
    }
}
