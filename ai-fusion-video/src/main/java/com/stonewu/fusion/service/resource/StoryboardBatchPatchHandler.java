package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.StoryboardMutationRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class StoryboardBatchPatchHandler implements ResourceCommandHandler {
    private final StoryboardCommandSupport support;
    private final StoryboardMutationRepository writes;
    private final StoryboardFields fields;
    private final StoryboardRelations relations;
    private final StoryboardVersionView versions;
    private final CanvasJson json;

    @Override
    public CanvasValueType resourceType() {
        return STORYBOARD;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.BATCH_PATCH_STORYBOARD;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        return support.authorize(STORYBOARD, ((StoryboardBatchPatchCommand) input).storyboardId(), context, false);
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (StoryboardBatchPatchCommand) input;
        var change = writes.lock(STORYBOARD, command.storyboardId());
        Set<String> seen = new HashSet<>();
        for (var entry : command.entries()) {
            if (!Set.of(STORYBOARD, STORYBOARD_EPISODE, STORYBOARD_SCENE, SHOT).contains(entry.resourceType())
                    || !seen.add(entry.resourceType().code() + ":" + entry.resourceId()) || entry.patch().isEmpty()) {
                throw StoryboardFields.invalid("批量修改只接受当前分镜中不重复的非空字段修改");
            }
            var header = change.header(entry.resourceType(), entry.resourceId());
            ResourceMutations.requireVersion(header, entry.expectedVersion());
            fields.values(entry.resourceType(), entry.patch(), false);
        }
        for (var entry : command.entries()) {
            relations.validate(change.header(STORYBOARD, command.storyboardId()), entry.resourceType(), entry.patch(),
                    entry.resourceId());
            change.patch(change.header(entry.resourceType(), entry.resourceId()),
                    fields.values(entry.resourceType(), entry.patch(), false));
        }
        change.finish();
        var results = command.entries().stream()
                .map(entry -> Map.of("resourceType", entry.resourceType(), "resourceId", entry.resourceId(),
                        "data", versions.attach(entry.resourceType(),
                                change.current(entry.resourceType(), entry.resourceId())))).toList();
        return change.result(json.object(Map.of("storyboardId", command.storyboardId(),
                "storyboardVersion", change.header(STORYBOARD, command.storyboardId()).version(), "resources",
                results)));
    }
}
