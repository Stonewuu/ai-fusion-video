package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.ScriptMutationRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class ScriptResourceCreateHandler implements ResourceCommandHandler {
    private final ScriptCommandSupport support;
    private final ScriptMutationRepository writes;
    private final ScriptFields fields;

    @Override
    public CanvasValueType resourceType() {
        return SCRIPT_EPISODE;
    }

    @Override
    public Set<CanvasValueType> resourceTypes() {
        return Set.of(SCRIPT_EPISODE, SCRIPT_SCENE);
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.CREATE;
    }

    @Override
    public void evict() {
        support.evict();
    }

    private CanvasValueType parentType(ResourceCreateCommand command) {
        return command.resourceType() == SCRIPT_EPISODE ? SCRIPT : SCRIPT_EPISODE;
    }

    private long parentId(ResourceCreateCommand command) {
        return ScriptFields.positiveId(
                command.fields().get(command.resourceType() == SCRIPT_EPISODE ? "scriptId" : "episodeId"), "父对象 ID");
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        var command = (ResourceCreateCommand) input;
        Long project = support.authorize(parentType(command), parentId(command), context, false);
        if (command.projectId() != null && !Objects.equals(project, command.projectId())) {
            throw ScriptFields.invalid("项目与父对象不一致");
        }
        return project;
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ResourceCreateCommand) input;
        int expected = ScriptFields.integer(command.fields().get("parentVersion"), 0, Integer.MAX_VALUE,
                "parentVersion");
        ObjectNode draft = command.fields().deepCopy();
        draft.remove(List.of(command.resourceType() == SCRIPT_EPISODE ? "scriptId" : "episodeId", "parentVersion"));
        Long afterId = null;
        if (command.resourceType() == SCRIPT_SCENE && draft.has("insertAfterId")) {
            afterId = ScriptFields.positiveId(draft.remove("insertAfterId"), "insertAfterId");
        }
        Map<String, Object> values = fields.values(command.resourceType(), draft);
        var change = writes.lock(parentType(command), parentId(command));
        ResourceMutations.requireVersion(change.target, expected);
        change.validateAssets(draft);
        values.put("scriptId", change.script.resourceId());
        ObjectNode result;
        if (command.resourceType() == SCRIPT_EPISODE) {
            int number = ScriptFields.integer(draft.get("episodeNumber"), 1, 1_000_000, "episodeNumber");
            change.requireEpisodeNumber(number, null);
            values.putIfAbsent("sortOrder", number);
            result = change.create(SCRIPT_EPISODE, values);
        } else {
            var existing = change.children(SCRIPT_SCENE, change.episode.resourceId());
            int index = values.containsKey("sortOrder") ? (Integer) values.get("sortOrder") : existing.size();
            if (index > existing.size()) {
                throw ScriptFields.invalid("插入位置超出当前场次列表");
            }
            if (afterId != null) {
                if (values.containsKey("sortOrder")) {
                    throw ScriptFields.invalid("不能同时指定两种插入位置");
                }
                index = -1;
                for (int i = 0; i < existing.size(); i++) {
                    if (existing.get(i).resourceId() == afterId) {
                        index = i + 1;
                    }
                }
                if (index < 0) {
                    throw ScriptFields.invalid("插入位置不属于当前分集");
                }
            }
            for (int i = 0; i < existing.size(); i++) {
                int order = i < index ? i : i + 1;
                var header = existing.get(i);
                if (change.data(header).path("sortOrder").intValue() != order) {
                    change.patch(header, Map.of("sortOrder", order));
                }
            }
            values.put("episodeId", change.episode.resourceId());
            values.put("sortOrder", index);
            values.putIfAbsent("sceneNumber",
                    change.data(change.episode).path("episodeNumber").intValue() + "-" + (index + 1));
            result = change.create(SCRIPT_SCENE, values);
            change.refreshEpisode();
        }
        change.refreshScript();
        return change.result(result);
    }
}
