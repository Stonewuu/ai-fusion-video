package com.stonewu.fusion.service.ai.tool.script;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.ScriptFields;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class ScriptToolQueries {
    private final ResourceQueryService queries;
    private final CanvasJson json;

    public String execute(String operation, String input, ToolExecutionContext context) {
        try {
            ObjectNode params = json.read(input, ObjectNode.class);
            var actor = context.requireMutationContext().actor();
            ObjectNode result = switch (operation) {
                case "get_script" -> queries.get(SCRIPT, id(params, "scriptId"), actor).data();
                case "get_script_scene" -> queries.get(SCRIPT_SCENE, id(params, "scriptSceneItemId"), actor).data();
                case "get_script_episode" -> queries.scriptTree(SCRIPT_EPISODE, id(params, "scriptEpisodeId"), actor);
                case "get_project_script", "get_script_structure" ->
                        queries.projectScriptTree(id(params, "projectId"), operation.equals("get_script_structure"),
                                actor);
                default -> throw ScriptFields.invalid("未登记的剧本查询");
            };
            if (result == null) {
                return json.write(Map.of("status", "empty", "message", "项目暂无剧本"));
            }
            aliases(result);
            String detail = params.path("detailLevel")
                    .asText(operation.equals("get_script_structure") ? "outline" : "summary");
            if (operation.equals("get_script_episode")) {
                if (!Set.of("summary", "full", "scenes_only").contains(detail)) {
                    throw ScriptFields.invalid("详细程度无效");
                }
                if (detail.equals("scenes_only")) {
                    JsonNode ids = params.get("scriptSceneItemIds");
                    if (ids == null || !ids.isArray() || ids.isEmpty()) {
                        throw ScriptFields.invalid("必须指定场次范围");
                    }
                    Set<Long> selected = new HashSet<>();
                    ids.forEach(value -> selected.add(ScriptFields.positiveId(value, "scriptSceneItemIds")));
                    ArrayNode scenes = (ArrayNode) result.get("scenes");
                    Set<Long> current = new HashSet<>();
                    scenes.forEach(scene -> current.add(scene.path("id").longValue()));
                    if (!current.containsAll(selected)) {
                        throw ScriptFields.invalid("所选场次不属于此分集或已失效");
                    }
                    for (int i = scenes.size() - 1; i >= 0; i--) {
                        if (!selected.contains(scenes.get(i).path("id").longValue())) {
                            scenes.remove(i);
                        }
                    }
                }
                if (detail.equals("summary")) {
                    trim(result, "summary");
                }
            }
            if (operation.equals("get_script_structure")) {
                if (!Set.of("outline", "summary", "full").contains(detail)) {
                    throw ScriptFields.invalid("详细程度无效");
                }
                if (!detail.equals("full")) {
                    trim(result, detail);
                }
            }
            result.put("status", "success");
            return json.write(result);
        } catch (CanvasException failure) {
            return json.write(
                    Map.of("status", "error", "message", failure.getMessage(), "errorCode", failure.getErrorCode(),
                            "details", failure.getDetails()));
        } catch (RuntimeException failure) {
            return json.write(Map.of("status", "error", "message",
                    failure.getMessage() == null ? "剧本查询失败" : failure.getMessage()));
        }
    }

    private long id(ObjectNode params, String name) {
        return ScriptFields.positiveId(params.get(name), name);
    }

    private void aliases(ObjectNode item) {
        if (item.has("episodeId")) {
            item.put("scriptSceneItemId", item.path("id").longValue())
                    .put("scriptEpisodeId", item.path("episodeId").longValue());
        } else if (item.has("scriptId")) {
            item.put("scriptEpisodeId", item.path("id").longValue())
                    .put("episode_version", item.path("version").intValue());
        } else {
            item.put("scriptId", item.path("id").longValue()).put("scriptVersion", item.path("version").intValue());
        }
        for (String key : List.of("episodes", "scenes")) {
            for (JsonNode child : item.path(key)) {
                aliases((ObjectNode) child);
            }
        }
    }

    private void trim(ObjectNode item, String detail) {
        item.remove(List.of("rawContent", "content", "dialogues"));
        if (detail.equals("outline")) {
            item.remove(List.of("synopsis", "storySynopsis", "charactersJson", "characters", "sceneDescription"));
        }
        for (String key : List.of("episodes", "scenes")) {
            for (JsonNode child : item.path(key)) {
                trim((ObjectNode) child, detail);
            }
        }
    }
}
