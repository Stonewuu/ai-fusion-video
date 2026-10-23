package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ScriptMutationRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.SCRIPT_EPISODE;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.SCRIPT_SCENE;

@Component
@RequiredArgsConstructor
public class ScriptScenesHandler implements ResourceCommandHandler {
    private final ScriptCommandSupport support;
    private final ScriptMutationRepository writes;
    private final ScriptFields fields;
    private final CanvasJson json;

    @Override
    public CanvasValueType resourceType() {
        return SCRIPT_EPISODE;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.MANAGE_SCRIPT_SCENES;
    }

    @Override
    public void evict() {
        support.evict();
    }

    @Override
    public Long authorize(ResourceCommand input, ResourceCommandContext context, ResourceCommandReceipt previous) {
        return support.authorize(SCRIPT_EPISODE, ((ScriptScenesCommand) input).resourceId(), context, false);
    }

    @Override
    public ResourceMutation apply(ResourceCommand input, ResourceCommandContext context) {
        var command = (ScriptScenesCommand) input;
        var change = writes.lock(SCRIPT_EPISODE, command.resourceId());
        ResourceMutations.requireVersion(change.episode, command.expectedVersion());
        var existing = change.children(SCRIPT_SCENE, command.resourceId());
        List<Long> ids = existing.stream().map(item -> item.resourceId()).toList();
        if (new HashSet<>(command.sceneIds()).size() != command.sceneIds().size()) {
            throw ScriptFields.invalid("场次 ID 不能重复");
        }
        switch (command.mode()) {
            case APPEND, REPLACE -> {
                if (!command.sceneIds()
                        .isEmpty() || (command.mode() == ScriptScenesCommand.Mode.APPEND && command.scenes()
                        .isEmpty())) {
                    throw ScriptFields.invalid("新增/替换模式仅接收 scenes");
                }
                var rows = command.scenes().stream().map(draft -> fields.values(SCRIPT_SCENE, draft)).toList();
                command.scenes().forEach(change::validateAssets);
                boolean replace = command.mode() == ScriptScenesCommand.Mode.REPLACE;
                if (replace) {
                    existing.forEach(change::delete);
                } else {
                    for (int i = 0; i < existing.size(); i++) {
                        if (change.data(existing.get(i)).path("sortOrder").intValue() != i) {
                            change.patch(existing.get(i), Map.of("sortOrder", i));
                        }
                    }
                }
                int start = replace ? 0 : existing.size();
                for (int i = 0; i < rows.size(); i++) {
                    var row = rows.get(i);
                    row.put("scriptId", change.script.resourceId());
                    row.put("episodeId", change.episode.resourceId());
                    row.put("sortOrder", start + i);
                    row.putIfAbsent("sceneNumber",
                            change.data(change.episode).path("episodeNumber").intValue() + "-" + (start + i + 1));
                    change.create(SCRIPT_SCENE, row);
                }
            }
            case DELETE -> {
                if (!command.scenes().isEmpty() || command.sceneIds().isEmpty() || !ids.containsAll(
                        command.sceneIds())) {
                    throw ScriptFields.invalid("删除场次必须全部属于当前分集");
                }
                int order = 0;
                for (var scene : existing) {
                    if (command.sceneIds().contains(scene.resourceId())) {
                        change.delete(scene);
                    } else {
                        if (change.data(scene).path("sortOrder").intValue() != order) {
                            change.patch(scene, Map.of("sortOrder", order));
                        }
                        order++;
                    }
                }
            }
            case REORDER -> {
                if (!command.scenes().isEmpty() || !new HashSet<>(ids).equals(new HashSet<>(command.sceneIds()))) {
                    throw ScriptFields.invalid("排序必须包含当前分集的全部场次且不能重复");
                }
                for (int i = 0; i < command.sceneIds().size(); i++) {
                    var scene = change.header(SCRIPT_SCENE, command.sceneIds().get(i));
                    if (change.data(scene).path("sortOrder").intValue() != i) {
                        change.patch(scene, Map.of("sortOrder", i));
                    }
                }
            }
        }
        change.refreshEpisode();
        change.refreshScript();
        var episode = change.data(change.header(SCRIPT_EPISODE, command.resourceId()));
        var scenes = change.children(SCRIPT_SCENE, command.resourceId()).stream().map(change::data).toList();
        return change.result(json.object(Map.of("episode", episode, "scenes", scenes, "mode", command.mode())));
    }
}
