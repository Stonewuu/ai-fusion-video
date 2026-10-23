package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

@Component
@RequiredArgsConstructor
public class ScriptFields {
    private final CanvasJson json;
    private static final Set<String> SCRIPT = Set.of("content", "rawContent", "storySynopsis", "charactersJson",
            "sourceType",
            "parsingStatus", "parsingProgress", "summary", "genre", "targetAudience", "durationEstimate", "status");
    private static final Set<String> EPISODE = Set.of("episodeNumber", "title", "synopsis", "rawContent",
            "durationEstimate",
            "sourceType", "sortOrder", "parsingStatus", "status");
    private static final Set<String> SCENE = Set.of("sceneNumber", "sceneHeading", "location", "timeOfDay", "intExt",
            "characters",
            "characterAssetIds", "sceneAssetId", "propAssetIds", "sceneDescription", "dialogues", "sortOrder",
            "status");
    private static final Set<String> JSON_ARRAYS = Set.of("charactersJson", "characters", "characterAssetIds",
            "propAssetIds", "dialogues");
    private static final Map<String, Integer> TEXT_LIMITS = Map.of("title", 256, "parsingProgress", 256, "genre", 128,
            "targetAudience", 128, "sceneNumber", 32, "sceneHeading", 256, "location", 128, "timeOfDay", 32, "intExt",
            16);
    private static final Set<String> NONNULL = Set.of("episodeNumber", "sortOrder", "status", "sourceType",
            "parsingStatus");

    public Map<String, Object> values(CanvasValueType type, ObjectNode input) {
        Set<String> allowed = switch (type) {
            case SCRIPT -> SCRIPT;
            case SCRIPT_EPISODE -> EPISODE;
            case SCRIPT_SCENE -> SCENE;
            default -> throw invalid("不支持的剧本对象类型");
        };
        Map<String, Object> result = new LinkedHashMap<>();
        input.fields().forEachRemaining(entry -> {
            String field = entry.getKey();
            JsonNode value = entry.getValue();
            if (!allowed.contains(field)) {
                throw invalid("不允许修改字段：" + field);
            }
            if (value.isNull()) {
                if (NONNULL.contains(field)) {
                    throw invalid(field + " 不能清空");
                }
                result.put(field, null);
            } else if (JSON_ARRAYS.contains(field)) {
                validateArray(field, value);
                result.put(field, json.write(value));
            } else if (Set.of("episodeNumber", "sortOrder", "durationEstimate", "status", "sourceType", "parsingStatus")
                    .contains(field)) {
                int min = field.equals("episodeNumber") ? 1 : 0;
                int max = switch (field) {
                    case "status" -> 1;
                    case "sourceType" -> 2;
                    case "parsingStatus" -> 3;
                    default -> 1_000_000;
                };
                result.put(field, integer(value, min, max, field));
            } else if (field.equals("sceneAssetId")) {
                result.put(field, positiveId(value, field));
            } else {
                if (!value.isTextual()) {
                    throw invalid(field + " 必须是文本");
                }
                String text = value.textValue();
                if (TEXT_LIMITS.containsKey(field) ? text.codePointCount(0, text.length()) > TEXT_LIMITS.get(field)
                        : text.getBytes(StandardCharsets.UTF_8).length > (field.equals(
                        "rawContent") ? 8_000_000 : 65_535)) {
                    throw invalid(field + " 内容过长");
                }
                result.put(field, text);
            }
        });
        return result;
    }

    private void validateArray(String field, JsonNode value) {
        if (!value.isArray() || value.size() > 10_000 || json.write(value)
                .getBytes(StandardCharsets.UTF_8).length > 1_048_576) {
            throw invalid(field + " 必须是长度受限的数组");
        }
        for (JsonNode item : value) {
            switch (field) {
                case "characters" -> {
                    if (!item.isTextual() || item.textValue().length() > 256) {
                        throw invalid("角色名称无效");
                    }
                }
                case "characterAssetIds", "propAssetIds" -> positiveId(item, field);
                case "charactersJson" -> {
                    if (!item.isObject() || !item.path("name").isTextual() || item.path("name").textValue().isBlank()) {
                        throw invalid("人物表必须包含姓名");
                    }
                    if (item.hasNonNull("assetId")) {
                        positiveId(item.get("assetId"), "assetId");
                    }
                }
                case "dialogues" -> {
                    if (!item.isObject()) {
                        throw invalid("对白元素必须为对象");
                    }
                    integer(item.path("type"), 1, 5, "type");
                    if (!item.path("content").isTextual()) {
                        throw invalid("对白元素必须包含 content");
                    }
                    if (item.hasNonNull("character_asset_id")) {
                        positiveId(item.get("character_asset_id"), "character_asset_id");
                    }
                    for (String key : Set.of("character_name", "parenthetical")) {
                        if (item.hasNonNull(key) && !item.get(key).isTextual()) {
                            throw invalid("对白 " + key + " 必须是文本");
                        }
                    }
                    if (item.hasNonNull("sortOrder")) {
                        integer(item.get("sortOrder"), 0, 1_000_000, "sortOrder");
                    }
                }
                default -> throw invalid("无效数组字段");
            }
        }
    }

    public static int integer(JsonNode value, int min, int max, String name) {
        if (value == null || !value.isIntegralNumber() || !value.canConvertToInt() || value.intValue() < min || value.intValue() > max) {
            throw invalid(name + " 必须为 " + min + " 至 " + max + " 的整数");
        }
        return value.intValue();
    }

    public static long positiveId(JsonNode value, String name) {
        if (value == null || !value.isIntegralNumber() || !value.canConvertToLong() || value.longValue() <= 0) {
            throw invalid(name + " 必须为正整数");
        }
        return value.longValue();
    }

    public static String column(String field) {
        if (!SCRIPT.contains(field) && !EPISODE.contains(field) && !SCENE.contains(field)
                && !Set.of("scriptId", "episodeId", "totalEpisodes", "totalScenes", "deleted").contains(field)) {
            throw invalid("字段未登记");
        }
        return field.replaceAll("([A-Z])", "_$1").toLowerCase(Locale.ROOT);
    }

    public static CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_COMMAND, message);
    }
}
