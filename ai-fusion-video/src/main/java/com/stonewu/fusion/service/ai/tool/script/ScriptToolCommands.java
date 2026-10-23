package com.stonewu.fusion.service.ai.tool.script;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceToolRequests;
import com.stonewu.fusion.service.resource.ScriptFields;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class ScriptToolCommands {
    private final ResourceCommandService commands;
    private final ResourceQueryService queries;
    private final CanvasJson json;

    public String patch(CanvasValueType type, String idField, String input, ToolExecutionContext context) {
        return invoke(() -> {
            var params = json.read(input, ObjectNode.class);
            long id = ScriptFields.positiveId(params.remove(idField), idField);
            int version = version(params.remove("expectedVersion"), "expectedVersion");
            var actor = context.requireMutationContext();
            return commands.patch(
                    new ResourcePatchCommand(ResourceToolRequests.requestId(json, actor), type, id, version, params),
                    actor);
        });
    }

    public String saveEpisode(String input, ToolExecutionContext context) {
        return invoke(() -> {
            var fields = json.read(input, ObjectNode.class);
            long scriptId = ScriptFields.positiveId(fields.remove("scriptId"), "scriptId");
            var actor = context.requireMutationContext();
            String requestId = ResourceToolRequests.requestId(json, actor);
            if (fields.has("scriptEpisodeId")) {
                long id = ScriptFields.positiveId(fields.remove("scriptEpisodeId"), "scriptEpisodeId");
                int expected = version(fields.remove("expectedVersion"), "expectedVersion");
                var existing = queries.get(SCRIPT_EPISODE, id, actor.actor()).data();
                if (existing.path("scriptId").longValue() != scriptId) {
                    throw ScriptFields.invalid("分集不属于指定剧本");
                }
                return commands.patch(new ResourcePatchCommand(requestId, SCRIPT_EPISODE, id, expected, fields), actor);
            }
            int expected = version(fields.remove("expectedVersion"), "expectedVersion");
            fields.put("scriptId", scriptId).put("parentVersion", expected);
            return commands.execute(new ResourceCreateCommand(requestId, SCRIPT_EPISODE, null, fields), actor);
        });
    }

    public String saveScenes(String input, ToolExecutionContext context) {
        return invoke(() -> {
            var params = json.read(input, ObjectNode.class);
            var scenes = scenes(params.get("scenes"));
            if (scenes.isEmpty() || scenes.size() > 2) {
                throw ScriptFields.invalid("每次必须提交 1 至 2 个场次");
            }
            if (params.has("overwriteMode") && !params.get("overwriteMode").isBoolean()) {
                throw ScriptFields.invalid("overwriteMode 必须为布尔值");
            }
            var mode = params.path("overwriteMode")
                    .asBoolean(false) ? ScriptScenesCommand.Mode.REPLACE : ScriptScenesCommand.Mode.APPEND;
            return sceneCommand(params, mode, scenes, List.of(), context);
        });
    }

    public String manageScenes(String input, ToolExecutionContext context) {
        return invoke(() -> {
            var params = json.read(input, ObjectNode.class);
            return switch (params.path("action").asText()) {
                case "add" ->
                        sceneCommand(params, ScriptScenesCommand.Mode.APPEND, scenes(params.get("scenes")), List.of(),
                                context);
                case "delete" -> {
                    JsonNode inputIds = params.get("scriptSceneItemIds");
                    if (inputIds == null || !inputIds.isArray() || inputIds.isEmpty()) {
                        throw ScriptFields.invalid("必须指定待删除场次 ID");
                    }
                    List<Long> ids = new ArrayList<>();
                    for (JsonNode id : inputIds) {
                        ids.add(ScriptFields.positiveId(id, "scriptSceneItemIds"));
                    }
                    yield sceneCommand(params, ScriptScenesCommand.Mode.DELETE, List.of(), ids, context);
                }
                default -> throw ScriptFields.invalid("action 仅支持 add 或 delete");
            };
        });
    }

    public String delete(String input, ToolExecutionContext context) {
        return invoke(() -> {
            var params = json.read(input, ObjectNode.class);
            CanvasValueType type = switch (params.path("resourceType").asText()) {
                case "episode" -> SCRIPT_EPISODE;
                case "scene" -> SCRIPT_SCENE;
                default -> throw ScriptFields.invalid("只能删除剧本分集或场次");
            };
            var actor = context.requireMutationContext();
            return commands.execute(new ResourceDeleteCommand(ResourceToolRequests.requestId(json, actor), type,
                    ScriptFields.positiveId(params.get("resourceId"), "resourceId"),
                    version(params.get("expectedVersion"), "expectedVersion")), actor);
        });
    }

    private ResourceCommandReceipt sceneCommand(ObjectNode params, ScriptScenesCommand.Mode mode,
                                                List<ObjectNode> scenes,
                                                List<Long> ids, ToolExecutionContext context) {
        var actor = context.requireMutationContext();
        return commands.execute(new ScriptScenesCommand(ResourceToolRequests.requestId(json, actor),
                ScriptFields.positiveId(params.get("scriptEpisodeId"), "scriptEpisodeId"),
                version(params.get("episode_version"), "episode_version"),
                mode, scenes, ids), actor);
    }

    private List<ObjectNode> scenes(JsonNode input) {
        if (input == null || !input.isArray()) {
            throw ScriptFields.invalid("scenes 必须是数组");
        }
        Map<String, String> aliases = Map.of("scene_heading", "sceneHeading", "time_of_day", "timeOfDay", "int_ext",
                "intExt",
                "character_asset_ids", "characterAssetIds", "scene_asset_id", "sceneAssetId", "prop_asset_ids",
                "propAssetIds", "scene_description", "sceneDescription");
        List<ObjectNode> result = new ArrayList<>();
        for (JsonNode value : input) {
            if (!(value instanceof ObjectNode object)) {
                throw ScriptFields.invalid("场次必须为对象");
            }
            ObjectNode row = json.object(Map.of());
            object.fields().forEachRemaining(field -> {
                String key = aliases.getOrDefault(field.getKey(), field.getKey());
                if (row.has(key)) {
                    throw ScriptFields.invalid("重复场次字段：" + key);
                }
                row.set(key, field.getValue());
            });
            result.add(row);
        }
        return result;
    }

    private int version(JsonNode value, String field) {
        return ScriptFields.integer(value, 0, Integer.MAX_VALUE, field);
    }

    private String invoke(Supplier<ResourceCommandReceipt> action) {
        try {
            var receipt = action.get();
            ObjectNode result = receipt.result().deepCopy();
            result.put("status", "success").put("commandId", receipt.commandId());
            result.set("changes", json.object(Map.of("changes", receipt.changes())).get("changes"));
            for (var change : receipt.changes()) {
                if (change.resourceType() == SCRIPT) {
                    result.put("scriptId", change.resourceId()).put("scriptVersion", change.version());
                }
                if (change.resourceType() == SCRIPT_EPISODE) {
                    result.put("scriptEpisodeId", change.resourceId()).put("episode_version", change.version());
                }
            }
            return json.write(result);
        } catch (CanvasException failure) {
            return json.write(
                    Map.of("status", "error", "message", failure.getMessage(), "errorCode", failure.getErrorCode(),
                            "details", failure.getDetails()));
        } catch (RuntimeException failure) {
            return json.write(Map.of("status", "error", "message",
                    failure.getMessage() == null ? "剧本命令失败" : failure.getMessage()));
        }
    }
}
