package com.stonewu.fusion.service.ai.tool.storyboard;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.*;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class StoryboardToolCommands {
    private final ResourceCommandService commands;
    private final ResourceQueryService resources;
    private final CanvasJson json;

    public String execute(String operation, String input, ToolExecutionContext invocation) {
        try {
            var actor = invocation.requireMutationContext();
            ObjectNode fields = json.read(input, ObjectNode.class);
            if (fields.has("requestId")) {
                throw StoryboardFields.invalid("工具请求标识由当前运行确定");
            }
            String request = ResourceToolRequests.requestId(json, actor);
            ResourceCommand command = switch (operation) {
                case "save_storyboard_episode" -> episode(fields, request, actor);
                case "save_storyboard_scene_shots" -> scene(fields, request, actor);
                case "insert_storyboard_item" -> {
                    requireText(fields.get("content"), "content");
                    int version = version(fields.remove("expectedParentVersion"), "expectedParentVersion");
                    fields.put("parentVersion", version).put("aiGenerated", true);
                    yield new ResourceCreateCommand(request, SHOT, null, fields);
                }
                case "update_storyboard_item", "update_storyboard_scene" -> {
                    boolean scene = operation.equals("update_storyboard_scene");
                    long id = StoryboardFields.id(fields.remove(scene ? "storyboardSceneId" : "storyboardItemId"),
                            "对象 ID");
                    int version = version(fields.remove("expectedVersion"), "expectedVersion");
                    yield new ResourcePatchCommand(request, scene ? STORYBOARD_SCENE : SHOT, id, version, fields);
                }
                case "update_storyboard_item_frame" -> frame(fields, request);
                case "update_storyboard_item_video" -> video(fields, request, actor);
                case "delete_storyboard_child_resource" -> {
                    CanvasValueType type = switch (fields.path("resourceType").asText()) {
                        case "episode" -> STORYBOARD_EPISODE;
                        case "scene" -> STORYBOARD_SCENE;
                        case "item" -> SHOT;
                        default -> throw StoryboardFields.invalid("只允许删除分镜集、场次或镜头");
                    };
                    requireOnly(fields, Set.of("resourceType", "resourceId", "expectedVersion"));
                    yield new ResourceDeleteCommand(request, type,
                            StoryboardFields.id(fields.get("resourceId"), "resourceId"),
                            version(fields.get("expectedVersion"), "expectedVersion"));
                }
                case "manage_storyboard_children" -> {
                    fields.put("requestId", request);
                    yield json.read(json.write(fields), StoryboardChildrenCommand.class);
                }
                case "batch_update_storyboard" -> {
                    fields.put("requestId", request);
                    yield json.read(json.write(fields), StoryboardBatchPatchCommand.class);
                }
                default -> throw StoryboardFields.invalid("分镜写入工具未登记");
            };
            var receipt = commands.execute(command, actor);
            ObjectNode result = receipt.result().deepCopy();
            alias(command.resourceType(), result);
            if (result.get("parent") instanceof ObjectNode parent) {
                alias(command.resourceType(), parent);
            }
            if (result.has("status")) {
                result.set("resourceStatus", result.get("status"));
            }
            result.put("status", "success").put("commandId", receipt.commandId());
            result.set("changes", json.object(Map.of("changes", receipt.changes())).get("changes"));
            var createdShots = receipt.changes().stream()
                    .filter(change -> change.resourceType() == SHOT && change.version() == 0 && !change.deleted())
                    .map(change -> change.resourceId()).toList();
            if (!createdShots.isEmpty()) {
                result.set("createdShotIds", json.object(Map.of("ids", createdShots)).get("ids"));
                result.put("shotCount", createdShots.size());
            }
            return json.write(result);
        } catch (CanvasException failure) {
            return json.write(
                    Map.of("status", "error", "message", failure.getMessage(), "errorCode", failure.getErrorCode(),
                            "details", failure.getDetails()));
        } catch (RuntimeException failure) {
            return json.write(Map.of("status", "error", "message",
                    failure.getMessage() == null ? "分镜命令失败" : failure.getMessage()));
        }
    }

    private ResourceCommand episode(ObjectNode fields, String request, ResourceCommandContext actor) {
        long board = StoryboardFields.id(fields.remove("storyboardId"), "storyboardId");
        if (fields.has("storyboardEpisodeId")) {
            long id = StoryboardFields.id(fields.remove("storyboardEpisodeId"), "storyboardEpisodeId");
            int expected = version(fields.remove("expectedVersion"), "expectedVersion");
            if (fields.has("expectedParentVersion")) {
                throw StoryboardFields.invalid("修改已有分集只使用 expectedVersion");
            }
            if (resources.get(STORYBOARD_EPISODE, id, actor.actor()).data().path("storyboardId").longValue() != board) {
                throw StoryboardFields.invalid("分镜集不属于指定分镜");
            }
            return new ResourcePatchCommand(request, STORYBOARD_EPISODE, id, expected, fields);
        }
        if (fields.has("expectedVersion")) {
            throw StoryboardFields.invalid("新建分集使用 expectedParentVersion");
        }
        int expected = version(fields.remove("expectedParentVersion"), "expectedParentVersion");
        fields.put("storyboardId", board).put("parentVersion", expected);
        return new ResourceCreateCommand(request, STORYBOARD_EPISODE, null, fields);
    }

    private ResourceCommand scene(ObjectNode fields, String request, ResourceCommandContext actor) {
        long board = StoryboardFields.id(fields.remove("storyboardId"), "storyboardId");
        long episode = StoryboardFields.id(fields.remove("storyboardEpisodeId"), "storyboardEpisodeId");
        int expected = version(fields.remove("expectedParentVersion"), "expectedParentVersion");
        if (resources.get(STORYBOARD_EPISODE, episode, actor.actor()).data().path("storyboardId")
                .longValue() != board) {
            throw StoryboardFields.invalid("分镜集不属于指定分镜");
        }
        JsonNode shots = fields.get("shots");
        if (shots == null || !shots.isArray() || shots.isEmpty() || shots.size() > 500) {
            throw StoryboardFields.invalid("每个场次必须包含 1 至 500 个镜头");
        }
        for (JsonNode shot : shots) {
            if (!(shot instanceof ObjectNode item)) {
                throw StoryboardFields.invalid("镜头必须为对象");
            }
            requireText(item.get("content"), "镜头 content");
            item.put("aiGenerated", true);
        }
        fields.put("episodeId", episode).put("parentVersion", expected);
        return new ResourceCreateCommand(request, STORYBOARD_SCENE, null, fields);
    }

    private ResourceCommand frame(ObjectNode fields, String request) {
        long id = StoryboardFields.id(fields.remove("storyboardItemId"), "storyboardItemId");
        int expected = version(fields.remove("expectedVersion"), "expectedVersion");
        requireOnly(fields, Set.of("frameType", "imageUrl", "framePrompt"));
        String type = requireText(fields.get("frameType"), "frameType");
        if (!Set.of("first", "last").contains(type)) {
            throw StoryboardFields.invalid("frameType 仅支持 first 或 last");
        }
        requireText(fields.get("imageUrl"), "imageUrl");
        ObjectNode patch = fields.objectNode();
        patch.set(type + "FrameImageUrl", fields.get("imageUrl"));
        if (fields.has("framePrompt")) {
            patch.set(type + "FramePrompt", fields.get("framePrompt"));
        }
        return new ResourcePatchCommand(request, SHOT, id, expected, patch);
    }

    private ResourceCommand video(ObjectNode fields, String request, ResourceCommandContext actor) {
        long id = StoryboardFields.id(fields.remove("storyboardItemId"), "storyboardItemId");
        int expected = version(fields.remove("expectedVersion"), "expectedVersion");
        requireOnly(fields, Set.of("videoUrl", "videoPrompt", "coverUrl"));
        if (fields.isEmpty()) {
            throw StoryboardFields.invalid("至少指定视频、提示词或封面");
        }
        ObjectNode patch = fields.objectNode();
        if (fields.has("videoUrl")) {
            patch.set("generatedVideoUrl", fields.get("videoUrl"));
        }
        if (fields.has("videoPrompt")) {
            patch.set("videoPrompt", fields.get("videoPrompt"));
        }
        if (fields.has("coverUrl")) {
            var target = resources.get(SHOT, id, actor.actor());
            ResourceMutations.requireVersion(target.header(), expected);
            JsonNode previous = target.data().get("customData");
            ObjectNode metadata;
            if (previous == null || previous.isNull()) {
                metadata = fields.objectNode();
            } else if (previous instanceof ObjectNode object) {
                metadata = object.deepCopy();
            } else {
                throw StoryboardFields.invalid("镜头扩展属性不是对象");
            }
            metadata.set("videoPosterUrl", fields.get("coverUrl"));
            patch.set("customData", metadata);
        }
        return new ResourcePatchCommand(request, SHOT, id, expected, patch);
    }

    private int version(JsonNode value, String key) {
        return StoryboardFields.integer(value, 0, Integer.MAX_VALUE, key);
    }

    private String requireText(JsonNode value, String key) {
        if (value == null || !value.isTextual() || value.textValue().isBlank()) {
            throw StoryboardFields.invalid(key + " 必须为非空文本");
        }
        return value.textValue();
    }

    private void requireOnly(ObjectNode fields, Set<String> allowed) {
        fields.fieldNames().forEachRemaining(name -> {
            if (!allowed.contains(name)) {
                throw StoryboardFields.invalid("未知工具参数：" + name);
            }
        });
    }

    private void alias(CanvasValueType type, ObjectNode result) {
        if (!result.has("id")) {
            return;
        }
        String key = switch (type) {
            case STORYBOARD -> "storyboardId";
            case STORYBOARD_EPISODE -> "storyboardEpisodeId";
            case STORYBOARD_SCENE -> "storyboardSceneId";
            case SHOT -> "storyboardItemId";
            default -> throw StoryboardFields.invalid("不是分镜对象");
        };
        result.put(key, result.path("id").longValue());
    }
}
