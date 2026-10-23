package com.stonewu.fusion.repository.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper;
import com.stonewu.fusion.mapper.resource.ResourceMutationMapper.NewResource;
import org.springframework.stereotype.Repository;

import java.util.*;

/**
 * 独立副本的列白名单。新增数据库字段需要明确决定复制语义，不能整行透传。
 */
@Repository
@RequiredArgsConstructor
public class ProjectForkRepository {
    public static final List<CanvasValueType> ORDER = List.of(CanvasValueType.PROJECT, CanvasValueType.ASSET,
            CanvasValueType.ASSET_ITEM,
            CanvasValueType.SCRIPT, CanvasValueType.SCRIPT_EPISODE, CanvasValueType.SCRIPT_SCENE,
            CanvasValueType.STORYBOARD,
            CanvasValueType.STORYBOARD_EPISODE, CanvasValueType.STORYBOARD_SCENE, CanvasValueType.SHOT);
    private final ResourceMutationMapper mutations;
    private final CanvasJson json;

    public long write(CanvasValueType type, Long target, ObjectNode data) {
        Map<String, Object> values = new LinkedHashMap<>();
        for (String field : fields(type)) {
            if (data.has(field)) {
                values.put(column(field), parameter(data.get(field)));
            }
        }
        values.put("version", data.path("version").intValue());
        values.put("deleted", data.path("deleted").asBoolean(false));
        if (target != null) {
            if (mutations.replaceCopy(type, target, values) != 1) {
                throw new BusinessException(409, "新建副本对象已发生变化");
            }
            return target;
        }
        var row = new NewResource(type, values);
        mutations.insertCopy(row);
        long id = row.getId();
        if (type == CanvasValueType.STORYBOARD_EPISODE && data.path("deleted").asBoolean(false)) {
            mutations.markDeletedEpisodeCopy(id);
        }
        return id;
    }

    public static List<String> fields(CanvasValueType type) {
        String fields = switch (type) {
            case PROJECT ->
                    "name description coverUrl properties artStyle artStyleDescription artStyleImagePrompt artStyleImageUrl status scope ownerType ownerId";
            case SCRIPT ->
                    "projectId title content rawContent totalEpisodes storySynopsis charactersJson sourceType parsingStatus parsingProgress summary genre targetAudience durationEstimate scope ownerType ownerId aiGenerated status";
            case SCRIPT_EPISODE ->
                    "scriptId episodeNumber title synopsis rawContent durationEstimate totalScenes sourceType sortOrder parsingStatus status";
            case SCRIPT_SCENE ->
                    "episodeId scriptId sceneNumber sceneHeading location timeOfDay intExt characters characterAssetIds sceneAssetId propAssetIds sceneDescription dialogues sortOrder status";
            case STORYBOARD ->
                    "projectId scriptId title description customColumns scope ownerType ownerId totalDuration status";
            case STORYBOARD_EPISODE ->
                    "storyboardId scriptEpisodeId episodeNumber title synopsis sortOrder status deletedId composedVideoUrl composeStatus composeErrorMsg composedAt bgmTaskId bgmAudioUrl bgmSfxAudioUrl bgmErrorMsg";
            case STORYBOARD_SCENE ->
                    "episodeId storyboardId sceneNumber sceneHeading location timeOfDay intExt sortOrder status";
            case SHOT ->
                    "storyboardId storyboardEpisodeId storyboardSceneId sortOrder shotNumber autoShotNumber imageUrl referenceImageUrl videoUrl generatedImageUrl firstFrameImageUrl lastFrameImageUrl firstFramePrompt lastFramePrompt generatedVideoUrl videoPrompt shotType duration content sceneExpectation sound dialogue soundEffect music cameraMovement cameraAngle cameraEquipment focalLength transition characterIds sceneAssetItemId propIds remark customData aiGenerated status";
            case ASSET ->
                    "userId projectId type name description coverUrl properties tags sourceType aiPrompt ownerType ownerId status";
            case ASSET_ITEM -> "assetId itemType name imageUrl thumbnailUrl properties sortOrder sourceType aiPrompt";
            default -> throw new BusinessException(400, "不能复制此业务类型");
        };
        return List.of(fields.split(" "));
    }

    private String column(String field) {
        return field.replaceAll("([a-z])([A-Z])", "$1_$2").toLowerCase(Locale.ROOT);
    }

    private Object parameter(JsonNode value) {
        if (value == null || value.isNull()) {
            return null;
        }
        if (value.isContainerNode()) {
            return json.write(value);
        }
        if (value.isBoolean()) {
            return value.booleanValue();
        }
        if (value.isNumber()) {
            return value.numberValue();
        }
        return value.asText();
    }
}
