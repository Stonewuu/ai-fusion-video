package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.MediaReference;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class StoryboardFields {
    private final CanvasJson json;
    private static final Set<String> BOARD = Set.of("description", "customColumns", "status");
    private static final Set<String> EPISODE = Set.of("scriptEpisodeId", "episodeNumber", "title", "synopsis",
            "sortOrder", "status");
    private static final Set<String> SCENE = Set.of("sceneNumber", "sceneHeading", "location", "timeOfDay", "intExt",
            "sortOrder", "status");
    private static final Set<String> ITEM = Set.of("shotNumber", "shotType", "duration", "content", "sceneExpectation",
            "sound",
            "dialogue", "soundEffect", "music", "cameraMovement", "cameraAngle", "cameraEquipment", "focalLength",
            "transition",
            "characterIds", "sceneAssetItemId", "propIds", "remark", "customData", "aiGenerated", "status", "sortOrder",
            "imageUrl", "referenceImageUrl", "videoUrl", "generatedImageUrl", "firstFrameImageUrl", "lastFrameImageUrl",
            "firstFramePrompt", "lastFramePrompt", "generatedVideoUrl", "videoPrompt");
    private static final Set<String> INTERNAL = Set.of("storyboardId", "episodeId", "storyboardEpisodeId",
            "storyboardSceneId",
            "deleted", "deletedId", "totalDuration", "autoShotNumber", "composeJobId", "composeStatus",
            "composeErrorMsg",
            "composedVideoUrl", "composedAt", "bgmTaskId", "bgmAudioUrl", "bgmSfxAudioUrl", "bgmErrorMsg");
    private static final Map<String, Integer> LENGTHS = Map.ofEntries(
            Map.entry("title", 200), Map.entry("sceneNumber", 20), Map.entry("sceneHeading", 200),
            Map.entry("location", 200),
            Map.entry("timeOfDay", 20), Map.entry("intExt", 20), Map.entry("shotNumber", 32), Map.entry("shotType", 32),
            Map.entry("soundEffect", 512), Map.entry("music", 512), Map.entry("cameraMovement", 64),
            Map.entry("cameraAngle", 64), Map.entry("cameraEquipment", 64), Map.entry("focalLength", 64),
            Map.entry("transition", 64));

    public Map<String, Object> values(CanvasValueType type, ObjectNode input, boolean creating) {
        Set<String> allowed = switch (type) {
            case STORYBOARD -> BOARD;
            case STORYBOARD_EPISODE -> EPISODE;
            case STORYBOARD_SCENE -> SCENE;
            case SHOT -> ITEM;
            default -> throw invalid("不是分镜对象");
        };
        Map<String, Object> result = new LinkedHashMap<>();
        input.fields().forEachRemaining(entry -> {
            String key = entry.getKey();
            JsonNode value = entry.getValue();
            if (!allowed.contains(key)) {
                throw invalid("不允许修改分镜字段：" + key);
            }
            if (key.equals("sortOrder") && !creating) {
                throw invalid("排序请使用有版本的集合命令");
            }
            if (value.isNull()) {
                if (Set.of("status", "sortOrder", "episodeNumber", "aiGenerated", "characterIds", "propIds")
                        .contains(key)) {
                    throw invalid(key + " 不能为 null，清空关联请使用空数组");
                }
                result.put(key, null);
            } else if (key.equals("status") || key.equals("sortOrder") || key.equals("episodeNumber")) {
                result.put(key,
                        integer(value, key.equals("episodeNumber") ? 1 : 0, key.equals("status") ? 1 : 1_000_000, key));
            } else if (key.equals("scriptEpisodeId") || key.equals("sceneAssetItemId")) {
                result.put(key, id(value, key));
            } else if (key.equals("duration")) {
                if (!value.isNumber()) {
                    throw invalid("duration 必须为数字");
                }
                BigDecimal duration = value.decimalValue();
                if (duration.signum() < 0 || duration.compareTo(
                        new BigDecimal("86400")) > 0 || duration.stripTrailingZeros().scale() > 2) {
                    throw invalid("duration 范围为 0 至 86400 秒，最多两位小数");
                }
                result.put(key, duration);
            } else if (key.equals("aiGenerated")) {
                if (!value.isBoolean()) {
                    throw invalid("aiGenerated 必须为布尔值");
                }
                result.put(key, value.booleanValue());
            } else if (key.equals("characterIds") || key.equals("propIds")) {
                ids(value, key);
                result.put(key, json.write(value));
            } else if (key.equals("customData") || key.equals("customColumns")) {
                boolean columns = key.equals("customColumns");
                if ((columns ? !value.isArray() : !value.isObject()) || value.size() > 200
                        || json.write(value).getBytes(StandardCharsets.UTF_8).length > 262_144) {
                    throw invalid(key + " JSON 格式或长度无效");
                }
                if (columns) {
                    for (JsonNode column : value) {
                        if (!column.isObject()) {
                            throw invalid("自定义列必须是对象");
                        }
                    }
                }
                if (!columns && value.hasNonNull("videoPosterUrl")) {
                    media(value.get("videoPosterUrl"), "videoPosterUrl");
                }
                result.put(key, json.write(value));
            } else {
                result.put(key,
                        key.endsWith("Url") ? media(value, key) : text(value, LENGTHS.getOrDefault(key, 65_535), key));
            }
        });
        if (creating && type == STORYBOARD_EPISODE && !result.containsKey("episodeNumber")) {
            throw invalid("创建分镜集必须提供 episodeNumber");
        }
        return result;
    }

    private void ids(JsonNode value, String key) {
        if (!value.isArray() || value.size() > 500) {
            throw invalid(key + " 必须为最多 500 项的 ID 数组");
        }
        Set<Long> seen = new HashSet<>();
        for (JsonNode item : value) {
            if (!seen.add(id(item, key))) {
                throw invalid(key + " 不能重复");
            }
        }
    }

    public static long id(JsonNode node, String name) {
        if (node == null || !node.isIntegralNumber() || !node.canConvertToLong() || node.longValue() <= 0) {
            throw invalid(name + " 必须是正整数 ID");
        }
        return node.longValue();
    }

    public static int integer(JsonNode node, int min, int max, String name) {
        if (node == null || !node.isIntegralNumber() || !node.canConvertToInt() || node.intValue() < min || node.intValue() > max) {
            throw invalid(name + " 整数范围无效");
        }
        return node.intValue();
    }

    private static String text(JsonNode node, int max, String name) {
        if (!node.isTextual() || node.textValue().codePointCount(0, node.textValue().length()) > max
                || node.textValue().getBytes(StandardCharsets.UTF_8).length > 65_535) {
            throw invalid(name + " 文本格式或长度无效");
        }
        return node.textValue();
    }

    private static String media(JsonNode node, String name) {
        String value = text(node, 1024, name);
        if (MediaReference.managed(value)) {
            MediaReference.id(value);
            return value;
        }
        if (value.isBlank()) {
            return value;
        }
        try {
            URI uri = new URI(value);
            boolean stored = value.startsWith("/") && !value.startsWith("//") && uri.getScheme() == null;
            boolean remote = ("https".equalsIgnoreCase(uri.getScheme()) || "http".equalsIgnoreCase(uri.getScheme()))
                    && uri.getHost() != null && uri.getUserInfo() == null;
            if ((!stored && !remote) || value.chars().anyMatch(Character::isISOControl)) {
                throw invalid(name + " 必须是已上传路径或 HTTP(S) 地址");
            }
            return value;
        } catch (URISyntaxException failure) {
            throw invalid(name + " 地址无效");
        }
    }

    public static CanvasValueType parent(CanvasValueType type) {
        return switch (type) {
            case STORYBOARD_EPISODE -> STORYBOARD;
            case STORYBOARD_SCENE -> STORYBOARD_EPISODE;
            case SHOT -> STORYBOARD_SCENE;
            default -> throw invalid("此对象没有可创建的父集合");
        };
    }

    public static CanvasValueType child(CanvasValueType type) {
        return switch (type) {
            case STORYBOARD -> STORYBOARD_EPISODE;
            case STORYBOARD_EPISODE -> STORYBOARD_SCENE;
            case STORYBOARD_SCENE -> SHOT;
            default -> throw invalid("此对象没有分镜子集合");
        };
    }

    public static String parentKey(CanvasValueType type) {
        return switch (type) {
            case STORYBOARD_EPISODE -> "storyboardId";
            case STORYBOARD_SCENE -> "episodeId";
            case SHOT -> "storyboardSceneId";
            default -> throw invalid("此对象没有父集合字段");
        };
    }

    public static String column(String field) {
        if (!BOARD.contains(field) && !EPISODE.contains(field) && !SCENE.contains(field) && !ITEM.contains(
                field) && !INTERNAL.contains(field)) {
            throw invalid("分镜字段未登记");
        }
        return field.replaceAll("([A-Z])", "_$1").toLowerCase(Locale.ROOT);
    }

    public static CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_COMMAND, message);
    }
}
