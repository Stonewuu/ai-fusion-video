package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.repository.resource.ResourceTables;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceRestorePlanner.Change;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/**
 * 恢复检查针对权威业务关系；删除视图不会删除资源，恢复也不能破坏后来增加的业务引用。
 */
@Component
@RequiredArgsConstructor
public class ResourceRestoreRelations {
    private final ResourceReadRepository resources;
    private final ResourceMutationMapper mutations;
    private final ResourceAccessPolicy access;

    private record Link(CanvasValueType from, String field, String column, CanvasValueType to, boolean collection,
                        String nestedField) {
        private Link(CanvasValueType from, String field, String column, CanvasValueType to, boolean collection) {
            this(from, field, column, to, collection, null);
        }
    }

    private static final List<Link> LINKS = List.of(
            new Link(SCRIPT, "projectId", "project_id", PROJECT, false),
            new Link(ASSET, "projectId", "project_id", PROJECT, false),
            new Link(STORYBOARD, "projectId", "project_id", PROJECT, false),
            new Link(STORYBOARD, "scriptId", "script_id", SCRIPT, false),
            new Link(SCRIPT_EPISODE, "scriptId", "script_id", SCRIPT, false),
            new Link(SCRIPT_SCENE, "scriptId", "script_id", SCRIPT, false),
            new Link(SCRIPT_SCENE, "episodeId", "episode_id", SCRIPT_EPISODE, false),
            new Link(ASSET_ITEM, "assetId", "asset_id", ASSET, false),
            new Link(STORYBOARD_EPISODE, "storyboardId", "storyboard_id", STORYBOARD, false),
            new Link(STORYBOARD_EPISODE, "scriptEpisodeId", "script_episode_id", SCRIPT_EPISODE, false),
            new Link(STORYBOARD_SCENE, "storyboardId", "storyboard_id", STORYBOARD, false),
            new Link(STORYBOARD_SCENE, "episodeId", "episode_id", STORYBOARD_EPISODE, false),
            new Link(SHOT, "storyboardId", "storyboard_id", STORYBOARD, false),
            new Link(SHOT, "storyboardEpisodeId", "storyboard_episode_id", STORYBOARD_EPISODE, false),
            new Link(SHOT, "storyboardSceneId", "storyboard_scene_id", STORYBOARD_SCENE, false),
            new Link(SCRIPT_SCENE, "sceneAssetId", "scene_asset_id", ASSET, false),
            new Link(SCRIPT_SCENE, "characterAssetIds", "character_asset_ids", ASSET, true),
            new Link(SCRIPT_SCENE, "propAssetIds", "prop_asset_ids", ASSET, true),
            new Link(SHOT, "sceneAssetItemId", "scene_asset_item_id", ASSET_ITEM, false),
            new Link(SHOT, "characterIds", "character_ids", ASSET_ITEM, true),
            new Link(SHOT, "propIds", "prop_ids", ASSET_ITEM, true),
            new Link(SCRIPT, "charactersJson", "characters_json", ASSET, true, "assetId"),
            new Link(SCRIPT_SCENE, "dialogues", "dialogues", ASSET, true, "character_asset_id"));

    public void validate(List<Change> changes, CanvasActor actor) {
        Map<String, Change> changed = changes.stream()
                .collect(Collectors.toMap(item -> key(item.resourceType(), item.resourceId()), Function.identity()));
        for (Change change : changes) {
            if (change.proposed().path("deleted").asBoolean()) {
                if (!change.current().path("deleted").asBoolean()) {
                    rejectNewDependents(change, changed);
                }
                continue;
            }
            for (Link link : LINKS) {
                if (link.from() == change.resourceType()) {
                    JsonNode value = change.proposed().path(link.field());
                    if (value.isMissingNode() || value.isNull()) {
                        continue;
                    }
                    if (link.collection()) {
                        for (JsonNode item : value) {
                            JsonNode id = link.nestedField() == null ? item : item.path(link.nestedField());
                            if (!id.isMissingNode() && !id.isNull()) {
                                requireTarget(link, id.longValue(), change.projectId(), changed, actor);
                            }
                        }
                    } else {
                        requireTarget(link, value.longValue(), change.projectId(), changed, actor);
                    }
                }
            }
            // 分镜场次/镜头的多个父引用必须互相一致。
            ObjectNode data = change.proposed();
            if (change.resourceType() == SHOT && data.hasNonNull("storyboardSceneId")) {
                var scene = target(STORYBOARD_SCENE, data.path("storyboardSceneId").longValue(), changed);
                if (scene.path("storyboardId").longValue() != data.path("storyboardId").longValue()
                        || (data.hasNonNull("storyboardEpisodeId") && scene.path("episodeId").longValue() != data.path(
                        "storyboardEpisodeId").longValue())) {
                    throw conflict("镜头的分集、场次关系已变化");
                }
            }
            if (change.resourceType() == SCRIPT_SCENE) {
                var episode = target(SCRIPT_EPISODE, data.path("episodeId").longValue(), changed);
                if (episode.path("scriptId").longValue() != data.path("scriptId").longValue()) {
                    throw conflict("剧本场次的所属关系已变化");
                }
            }
            if (change.resourceType() == SHOT && data.hasNonNull("storyboardEpisodeId")
                    && target(STORYBOARD_EPISODE, data.path("storyboardEpisodeId").asLong(), changed).path(
                    "storyboardId").asLong() != data.path("storyboardId").asLong()) {
                throw conflict("镜头分集不属于对应分镜");
            }
            if (change.resourceType() == STORYBOARD_SCENE
                    && target(STORYBOARD_EPISODE, data.path("episodeId").asLong(), changed).path("storyboardId")
                    .asLong() != data.path("storyboardId").asLong()) {
                throw conflict("分镜场次的所属分集已变化");
            }
            if (change.resourceType() == STORYBOARD_EPISODE && data.hasNonNull("scriptEpisodeId")) {
                var episode = target(SCRIPT_EPISODE, data.path("scriptEpisodeId").asLong(), changed);
                var board = target(STORYBOARD, data.path("storyboardId").asLong(), changed);
                if (episode.path("scriptId").asLong() != board.path("scriptId").asLong()) {
                    throw conflict("绑定分集不属于当前分镜关联的剧本");
                }
            }
        }
    }

    private void rejectNewDependents(Change removed, Map<String, Change> changed) {
        for (Link link : LINKS) {
            if (link.to() == removed.resourceType()) {
                var ids = mutations.referencing(link.from(), link.column(), link.collection(),
                        link.nestedField() == null ? null : "$[*]." + link.nestedField(),
                        removed.resourceId(), Long.toString(removed.resourceId()));
                for (Long id : ids) {
                    var dependent = changed.get(key(link.from(), id));
                    if (dependent == null || (!dependent.proposed().path("deleted").asBoolean() && references(
                            dependent.proposed(), link, removed.resourceId()))) {
                        throw conflict("对象已被其他业务内容使用，不能通过恢复操作移除");
                    }
                }
            }
        }
    }

    private void requireTarget(Link link, long id, Long projectId, Map<String, Change> changed, CanvasActor actor) {
        var type = link.to();
        if (id <= 0 || target(type, id, changed).path("deleted").asBoolean()) {
            throw conflict("恢复所需的关联对象已删除，请先处理关联内容");
        }
        var proposed = changed.get(key(type, id));
        Long targetProject;
        if (proposed != null) {
            targetProject = proposed.projectId();
        } else {
            var header = resources.require(type, id, false);
            access.require(header, actor);
            targetProject = header.projectId();
        }
        if (!Objects.equals(projectId, targetProject)) {
            throw conflict("恢复所需的关联对象不属于原项目");
        }
        String expectedType = switch (link.field()) {
            case "sceneAssetId", "sceneAssetItemId" -> "scene";
            case "propAssetIds", "propIds" -> "prop";
            case "characterAssetIds", "characterIds", "charactersJson", "dialogues" -> "character";
            default -> null;
        };
        if (expectedType != null) {
            var asset = type == ASSET ? target(type, id, changed) : target(ASSET,
                    target(type, id, changed).path("assetId").asLong(), changed);
            if (!expectedType.equals(asset.path("type").asText())) {
                throw conflict("恢复所需的关联资产类型已变化");
            }
        }
    }

    private boolean references(ObjectNode data, Link link, long id) {
        JsonNode value = data.path(link.field());
        if (link.collection()) {
            for (JsonNode item : value) {
                if ((link.nestedField() == null ? item : item.path(link.nestedField())).asLong(-1) == id) {
                    return true;
                }
            }
            return false;
        }
        return value.asLong(-1) == id;
    }

    private ObjectNode target(CanvasValueType type, long id, Map<String, Change> changed) {
        var planned = changed.get(key(type, id));
        return planned == null ? resources.raw(type, id, false) : planned.proposed();
    }

    private String key(CanvasValueType type, long id) {
        return type.code() + ":" + id;
    }

    private CanvasException conflict(String message) {
        return new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT, message, Map.of());
    }
}
