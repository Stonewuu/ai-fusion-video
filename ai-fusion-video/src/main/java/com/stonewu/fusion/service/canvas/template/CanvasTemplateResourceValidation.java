package com.stonewu.fusion.service.canvas.template;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.resource.ProjectForkRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.AssetFields;
import com.stonewu.fusion.service.resource.ProjectFields;
import com.stonewu.fusion.service.resource.ScriptFields;
import com.stonewu.fusion.service.resource.StoryboardFields;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 文件中的正文重新经过现行业务字段校验；归属、任务租约和原库访问权不属于模板协议。
 */
@Component
@RequiredArgsConstructor
public class CanvasTemplateResourceValidation {
    public static final Set<String> OWNERSHIP = Set.of("ownerType", "ownerId", "userId", "scope");
    private final ProjectFields projects;
    private final ScriptFields scripts;
    private final AssetFields assets;
    private final StoryboardFields boards;

    public ObjectNode clean(CanvasValueType type, ObjectNode original) {
        ObjectNode data = original.deepCopy();
        Set<String> allowed = new HashSet<>(ProjectForkRepository.fields(type));
        allowed.addAll(List.of("id", "version", "deleted"));
        allowed.removeAll(OWNERSHIP);
        allowed.remove("bgmTaskId");
        List<String> removed = new ArrayList<>();
        data.fieldNames().forEachRemaining(name -> {
            if (!allowed.contains(name)) {
                removed.add(name);
            }
        });
        data.remove(removed);
        if (type == CanvasValueType.SHOT) {
            for (String field : List.of("characterIds", "propIds")) {
                if (!data.hasNonNull(field)) {
                    data.putArray(field);
                }
            }
        }
        return data;
    }

    public void validate(CanvasValueType type, long id, ObjectNode data) {
        if (!type.isBusinessResource() || data == null || id <= 0 || !data.path("id").isIntegralNumber() || data.path(
                "id").asLong(-1) != id) {
            throw invalid("模板业务对象的类型或编号无效");
        }
        var expected = clean(type, data);
        if (!expected.equals(data)) {
            throw invalid("模板业务正文含未声明字段、归属字段或未规范化的数组");
        }
        integer(data, "version", 0, Integer.MAX_VALUE - 1, false);
        if (!data.path("deleted").isBoolean()) {
            throw invalid("模板对象必须声明删除状态");
        }
        var fields = data.deepCopy();
        fields.remove(List.of("id", "version", "deleted"));
        switch (type) {
            case PROJECT -> projects.values(fields, true);
            case ASSET -> {
                fields.remove("projectId");
                assets.values(type, fields, true);
                if (fields.hasNonNull("properties")) {
                    assets.validateProperties(fields.path("type").asText(), fields.path("properties"));
                }
            }
            case ASSET_ITEM -> {
                requiredId(data, "assetId");
                fields.remove("assetId");
                // 持久化的子类型允许为空；模板保留原空值，不将其解释为编辑时的清空操作。
                if (!fields.hasNonNull("itemType")) {
                    fields.remove("itemType");
                }
                assets.values(type, fields, true);
            }
            case SCRIPT -> {
                requiredId(data, "projectId");
                text(data, "title", 256);
                integer(data, "totalEpisodes", 0, 10000, true);
                flag(data, "aiGenerated");
                fields.remove(List.of("projectId", "title", "totalEpisodes", "aiGenerated"));
                scripts.values(type, fields);
            }
            case SCRIPT_EPISODE -> {
                requiredId(data, "scriptId");
                integer(data, "episodeNumber", 1, 1000000, false);
                integer(data, "totalScenes", 0, 10000, true);
                fields.remove(List.of("scriptId", "totalScenes"));
                scripts.values(type, fields);
            }
            case SCRIPT_SCENE -> {
                requiredId(data, "scriptId");
                requiredId(data, "episodeId");
                fields.remove(List.of("scriptId", "episodeId"));
                scripts.values(type, fields);
            }
            case STORYBOARD -> {
                requiredId(data, "projectId");
                requiredId(data, "scriptId");
                text(data, "title", 256);
                integer(data, "totalDuration", 0, Integer.MAX_VALUE, true);
                fields.remove(List.of("projectId", "scriptId", "title", "totalDuration"));
                boards.values(type, fields, true);
            }
            case STORYBOARD_EPISODE -> {
                requiredId(data, "storyboardId");
                integer(data, "deletedId", 0, Integer.MAX_VALUE, true);
                integer(data, "composeStatus", 0, 3, true);
                for (String field : List.of("composeErrorMsg", "bgmErrorMsg")) {
                    text(data, field, 65535);
                }
                text(data, "bgmTaskId", 255);
                text(data, "composedAt", 64);
                if (data.hasNonNull("composedAt")) {
                    try {
                        LocalDateTime.parse(data.path("composedAt").asText());
                    } catch (DateTimeParseException failure) {
                        throw invalid("模板成片时间格式无效");
                    }
                }
                // 这些媒体字段另外由模板素材协议校验，不能携带任意外部地址。
                fields.remove(List.of("storyboardId", "deletedId", "composeStatus", "composeErrorMsg", "composedAt",
                        "bgmTaskId", "bgmAudioUrl", "bgmSfxAudioUrl", "bgmErrorMsg", "composedVideoUrl"));
                boards.values(type, fields, true);
            }
            case STORYBOARD_SCENE -> {
                requiredId(data, "storyboardId");
                requiredId(data, "episodeId");
                fields.remove(List.of("storyboardId", "episodeId"));
                boards.values(type, fields, true);
            }
            case SHOT -> {
                requiredId(data, "storyboardId");
                text(data, "autoShotNumber", 64);
                fields.remove(List.of("storyboardId", "storyboardEpisodeId", "storyboardSceneId", "autoShotNumber"));
                boards.values(type, fields, true);
            }
            default -> throw invalid("模板业务类型未登记");
        }
    }

    private void requiredId(ObjectNode data, String field) {
        ScriptFields.positiveId(data.get(field), field);
    }

    private void integer(ObjectNode data, String field, int min, int max, boolean optional) {
        if (optional && !data.hasNonNull(field)) {
            return;
        }
        ScriptFields.integer(data.get(field), min, max, field);
    }

    private void text(ObjectNode data, String field, int max) {
        JsonNode value = data.get(field);
        if (value == null || value.isNull()) {
            return;
        }
        if (!value.isTextual() || value.asText().length() > max) {
            throw invalid("模板字段格式或长度无效：" + field);
        }
    }

    private void flag(ObjectNode data, String field) {
        if (data.hasNonNull(field) && !data.get(field).isBoolean()) {
            throw invalid("模板字段必须为布尔值：" + field);
        }
    }

    private BusinessException invalid(String text) {
        return new BusinessException(400, text);
    }
}
