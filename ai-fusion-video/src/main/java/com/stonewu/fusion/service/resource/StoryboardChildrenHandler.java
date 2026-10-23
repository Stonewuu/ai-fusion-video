package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.StoryboardMutationRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class StoryboardChildrenHandler implements ResourceCommandHandler {
    private final StoryboardCommandSupport support;
    private final StoryboardMutationRepository writes;
    private final StoryboardCreation creation;
    private final StoryboardVersionView versions;
    private final CanvasJson json;

    @Override
    public CanvasValueType resourceType() {
        return STORYBOARD;
    }

    @Override
    public Set<CanvasValueType> resourceTypes() {
        return Set.of(STORYBOARD, STORYBOARD_EPISODE, STORYBOARD_SCENE);
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.MANAGE_STORYBOARD_CHILDREN;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        var command = (StoryboardChildrenCommand) input;
        return support.authorize(command.resourceType(), command.resourceId(), context, false);
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (StoryboardChildrenCommand) input;
        var change = writes.lock(command.resourceType(), command.resourceId());
        ResourceMutations.requireVersion(change.target, command.expectedVersion());
        var existing = change.children(change.target);
        var ids = existing.stream().map(ResourceHeader::resourceId).toList();
        if (new HashSet<>(command.childIds()).size() != command.childIds().size()) {
            throw StoryboardFields.invalid("子对象 ID 不能重复");
        }
        switch (command.mode()) {
            case APPEND, REPLACE -> {
                if (!command.childIds()
                        .isEmpty() || (command.mode() == StoryboardChildrenCommand.Mode.APPEND && command.children()
                        .isEmpty())) {
                    throw StoryboardFields.invalid("追加/重建模式仅接收 children");
                }
                boolean replace = command.mode() == StoryboardChildrenCommand.Mode.REPLACE;
                if (replace) {
                    change.clear(change.target);
                }
                var order = new ArrayList<>(replace ? List.<Long>of() : ids);
                for (var child : command.children()) {
                    order.add(creation.createAt(change, change.target, child, order.size()));
                }
                change.reorder(change.target, order);
            }
            case CLEAR -> {
                if (!command.children().isEmpty() || !command.childIds().isEmpty()) {
                    throw StoryboardFields.invalid("清空模式不接受子对象参数");
                }
                change.clear(change.target);
            }
            case DELETE -> {
                if (!command.children().isEmpty() || command.childIds().isEmpty() || !ids.containsAll(
                        command.childIds())) {
                    throw StoryboardFields.invalid("删除对象必须全部属于当前集合");
                }
                existing.stream().filter(item -> command.childIds().contains(item.resourceId()))
                        .forEach(change::deleteTree);
                change.reorder(change.target, ids.stream().filter(id -> !command.childIds().contains(id)).toList());
            }
            case REORDER -> {
                if (!command.children().isEmpty()) {
                    throw StoryboardFields.invalid("排序模式不接受 children");
                }
                change.reorder(change.target, command.childIds());
            }
        }
        change.finish();
        var children = change.children(change.header(command.resourceType(), command.resourceId())).stream()
                .map(item -> versions.attach(item.resourceType(), change.data(item))).toList();
        return change.result(json.object(Map.of("parent",
                versions.attach(command.resourceType(), change.current(command.resourceType(), command.resourceId())),
                "children", children, "mode", command.mode())));
    }
}
