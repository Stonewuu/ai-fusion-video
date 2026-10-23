package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.StoryboardMutationRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class StoryboardResourceCreateHandler implements ResourceCommandHandler {
    private final StoryboardCommandSupport support;
    private final StoryboardMutationRepository writes;
    private final StoryboardCreation creation;
    private final StoryboardVersionView versions;

    @Override
    public CanvasValueType resourceType() {
        return STORYBOARD_EPISODE;
    }

    @Override
    public Set<CanvasValueType> resourceTypes() {
        return Set.of(STORYBOARD_EPISODE, STORYBOARD_SCENE, SHOT);
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.CREATE;
    }

    @Override
    public void evict() {
        support.evict();
    }

    private long parentId(ResourceCreateCommand command) {
        return StoryboardFields.id(command.fields().get(StoryboardFields.parentKey(command.resourceType())),
                "父对象 ID");
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        var command = (ResourceCreateCommand) input;
        Long project = support.authorize(StoryboardFields.parent(command.resourceType()), parentId(command), context,
                false);
        if (command.projectId() != null && !Objects.equals(project, command.projectId())) {
            throw StoryboardFields.invalid("项目与父对象不一致");
        }
        return project;
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ResourceCreateCommand) input;
        int expected = StoryboardFields.integer(command.fields().get("parentVersion"), 0, Integer.MAX_VALUE,
                "parentVersion");
        var change = writes.lock(StoryboardFields.parent(command.resourceType()), parentId(command));
        ResourceMutations.requireVersion(change.target, expected);
        var draft = command.fields().deepCopy();
        draft.remove(List.of(StoryboardFields.parentKey(command.resourceType()), "parentVersion"));
        long id = creation.insert(change, change.target, draft);
        change.finish();
        return change.result(versions.attach(command.resourceType(), change.current(command.resourceType(), id)));
    }
}
