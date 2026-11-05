package com.stonewu.fusion.service.ai.tool.storyboard;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.mapper.resource.ResourceReadMapper.ShotCount;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardFields;
import com.stonewu.fusion.service.resource.StoryboardResourceQueryService;
import com.stonewu.fusion.service.system.SystemConfigService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class StoryboardToolQueries {
    private final StoryboardResourceQueryService boards;
    private final ResourceQueryService resources;
    private final SystemConfigService configuration;
    private final CanvasJson json;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public String execute(String operation, String input, ToolExecutionContext invocation) {
        try {
            ObjectNode params = json.read(input, ObjectNode.class);
            var actor = invocation.requireMutationContext().actor();
            ObjectNode result = switch (operation) {
                case "get_project_storyboard" -> {
                    var board = boards.project(StoryboardFields.id(params.get("projectId"), "projectId"), actor);
                    if (board == null) {
                        yield json.object(Map.of("hasStoryboard", false));
                    }
                    board.put("hasStoryboard", true).put("storyboardId", board.path("id").longValue());
                    yield board;
                }
                case "get_storyboard" -> details(params, actor);
                case "get_storyboard_scene_items" -> scene(params, actor);
                default -> throw StoryboardFields.invalid("分镜查询工具未登记");
            };
            if (result.has("status")) {
                result.set("resourceStatus", result.get("status"));
            }
            result.put("status", "success");
            return json.write(result);
        } catch (CanvasException failure) {
            return json.write(
                    Map.of("status", "error", "message", failure.getMessage(), "errorCode", failure.getErrorCode(),
                            "details", failure.getDetails()));
        } catch (RuntimeException failure) {
            return json.write(Map.of("status", "error", "message",
                    failure.getMessage() == null ? "分镜查询失败" : failure.getMessage()));
        }
    }

    private ObjectNode details(ObjectNode params, CanvasActor actor) {
        params.fieldNames().forEachRemaining(key -> {
            if (!Set.of("storyboardId", "storyboardEpisodeId").contains(key)) {
                throw StoryboardFields.invalid("分镜查询仅支持 storyboardId 和单个 storyboardEpisodeId");
            }
        });
        long id = StoryboardFields.id(params.get("storyboardId"), "storyboardId");
        if (!params.has("storyboardEpisodeId")) {
            return overview(id, actor);
        }
        long episodeId = StoryboardFields.id(params.get("storyboardEpisodeId"), "storyboardEpisodeId");
        ObjectNode board = resources.get(STORYBOARD, id, actor).data();
        ObjectNode episode = resources.get(STORYBOARD_EPISODE, episodeId, actor).data();
        if (episode.path("storyboardId").longValue() != id) {
            throw StoryboardFields.invalid("指定分集不属于当前分镜");
        }
        episode = boards.tree(STORYBOARD_EPISODE, episodeId, 1, actor);
        List<ObjectNode> items = boards.children(SHOT, STORYBOARD_EPISODE, episodeId, actor);
        episode.put("storyboardEpisodeId", episodeId).put("itemCount", items.size());
        media(episode);
        Map<Long, ObjectNode> sceneById = new HashMap<>();
        for (JsonNode scene : episode.path("scenes")) {
            ObjectNode entry = (ObjectNode) scene;
            entry.put("storyboardSceneId", entry.path("id").longValue());
            sceneById.put(entry.path("id").longValue(), entry);
        }
        for (ObjectNode item : items) {
            item.put("storyboardItemId", item.path("id").longValue());
            ObjectNode scene = sceneById.get(item.path("storyboardSceneId").longValue());
            if (scene == null) {
                item.put("hierarchyState", "unassigned");
            } else {
                item.set("sceneNumber", scene.get("sceneNumber"));
                item.set("sceneHeading", scene.get("sceneHeading"));
            }
            media(item);
        }
        board.put("storyboardId", id).put("storyboardEpisodeId", episodeId)
                .put("scope", "episode").put("totalItems", items.size());
        board.putArray("episodes").add(episode);
        board.set("items", json.object(Map.of("items", items)).get("items"));
        return board;
    }

    private ObjectNode overview(long id, CanvasActor actor) {
        ObjectNode board = boards.tree(STORYBOARD, id, 2, actor);
        List<ShotCount> counts = boards.shotCounts(id, actor);
        Map<Long, Long> episodeCounts = new HashMap<>();
        Map<Long, Long> sceneCounts = new HashMap<>();
        for (ShotCount count : counts) {
            if (count.episodeId() != null) {
                episodeCounts.merge(count.episodeId(), count.itemCount(), Long::sum);
            }
            if (count.sceneId() != null) {
                sceneCounts.merge(count.sceneId(), count.itemCount(), Long::sum);
            }
        }
        for (JsonNode episode : board.path("episodes")) {
            ObjectNode value = (ObjectNode) episode;
            long episodeId = value.path("id").longValue();
            value.put("storyboardEpisodeId", episodeId).put("itemCount", episodeCounts.getOrDefault(episodeId, 0L))
                    .put("sceneCount", value.path("scenes").size());
            for (JsonNode scene : value.path("scenes")) {
                ObjectNode entry = (ObjectNode) scene;
                long sceneId = entry.path("id").longValue();
                entry.put("storyboardSceneId", sceneId).put("itemCount", sceneCounts.getOrDefault(sceneId, 0L));
            }
            media(value);
        }
        board.put("storyboardId", id).put("scope", "overview")
                .put("totalItems", counts.stream().mapToLong(ShotCount::itemCount).sum())
                .put("unassignedItems", counts.stream().filter(count -> count.episodeId() == null)
                        .mapToLong(ShotCount::itemCount).sum());
        return board;
    }

    private ObjectNode scene(ObjectNode params, CanvasActor actor) {
        Long sceneId = params.has("storyboardSceneId") ? StoryboardFields.id(params.get("storyboardSceneId"),
                "storyboardSceneId") : null;
        Long targetId = params.has("storyboardItemId") ? StoryboardFields.id(params.get("storyboardItemId"),
                "storyboardItemId") : null;
        if (targetId != null) {
            var target = resources.get(SHOT, targetId, actor).data();
            if (!target.hasNonNull("storyboardSceneId")) {
                throw StoryboardFields.invalid("目标镜头尚未归入场次");
            }
            long actualScene = target.path("storyboardSceneId").longValue();
            if (sceneId != null && sceneId != actualScene) {
                throw StoryboardFields.invalid("目标镜头与指定场次不一致");
            }
            sceneId = actualScene;
        }
        if (sceneId == null) {
            throw StoryboardFields.invalid("必须提供场次或镜头 ID");
        }
        var header = resources.get(STORYBOARD_SCENE, sceneId, actor).header();
        ObjectNode result = boards.tree(STORYBOARD_SCENE, sceneId, 1, actor);
        Map<Long, ObjectNode> refs = new HashMap<>();
        for (JsonNode row : result.path("items")) {
            ObjectNode item = (ObjectNode) row;
            item.put("storyboardItemId", item.path("id").longValue());
            if (targetId != null) {
                item.put("isCurrentTarget", item.path("id").longValue() == targetId);
            }
            for (String key : List.of("characterIds", "propIds")) {
                var references = item.putArray(key.equals("characterIds") ? "characterRefs" : "propRefs");
                JsonNode ids = item.get(key);
                if (ids != null && !ids.isNull()) {
                    if (!ids.isArray()) {
                        throw StoryboardFields.invalid("镜头资产引用不是 ID 数组");
                    }
                    for (JsonNode id : ids) {
                        references.add(reference(StoryboardFields.id(id, key), header.projectId(), actor, refs));
                    }
                }
            }
            if (item.hasNonNull("sceneAssetItemId")) {
                item.set("sceneRef",
                        reference(item.path("sceneAssetItemId").longValue(), header.projectId(), actor, refs));
            }
            media(item);
        }
        result.put("storyboardSceneId", sceneId).put("sceneVersion", header.version())
                .put("totalItems", result.path("items").size());
        result.set("sceneName", result.get("sceneHeading"));
        return result;
    }

    private ObjectNode reference(long id, Long project, CanvasActor actor, Map<Long, ObjectNode> refs) {
        return refs.computeIfAbsent(id, ignored -> {
            try {
                var snapshot = resources.get(ASSET_ITEM, id, actor);
                if (!Objects.equals(snapshot.header().projectId(), project)) {
                    return json.object(
                            Map.of("assetItemId", id, "unavailable", true, "errorCode", "INVALID_RESOURCE_REFERENCE"));
                }
                ObjectNode result = snapshot.data();
                result.put("assetItemId", id);
                media(result);
                return result;
            } catch (CanvasException unavailable) {
                if (!List.of(CanvasErrorCode.RESOURCE_DELETED, CanvasErrorCode.ACCESS_DENIED)
                        .contains(unavailable.getErrorCode())) {
                    throw unavailable;
                }
                return json.object(
                        Map.of("assetItemId", id, "unavailable", true, "errorCode", unavailable.getErrorCode()));
            }
        });
    }

    private void media(ObjectNode value) {
        for (String key : List.of("imageUrl", "thumbnailUrl", "referenceImageUrl", "videoUrl", "generatedImageUrl",
                "firstFrameImageUrl", "lastFrameImageUrl", "generatedVideoUrl", "composedVideoUrl", "bgmAudioUrl",
                "bgmSfxAudioUrl")) {
            if (value.hasNonNull(key)) {
                String stored = value.path(key).textValue();
                value.put(key + "Stored", stored);
                value.put(key, (MediaReference.managed(stored) ? stored : configuration.resolvePublicUrl(stored)));
            }
        }
        if (value.get("customData") instanceof ObjectNode data && data.hasNonNull("videoPosterUrl")) {
            String stored = data.path("videoPosterUrl").textValue();
            data.put("videoPosterUrlStored", stored);
            data.put("videoPosterUrl",
                    (MediaReference.managed(stored) ? stored : configuration.resolvePublicUrl(stored)));
        }
    }
}
