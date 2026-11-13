package com.stonewu.fusion.service.canvas.run;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.asset.AssetMetadataRegistry;
import com.stonewu.fusion.service.asset.AssetMetadataRegistry.FieldDef;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** 镜头描述作为视频创作内容，资产只提供外观和环境设定。 */
public final class CanvasVideoPrompt {
    private record Field(String key, String label) {
    }

    private static final List<Field> SHOT_FIELDS = List.of(
            new Field("videoPrompt", "视频要求"),
            new Field("content", "画面内容"),
            new Field("sceneExpectation", "场景要求"),
            new Field("dialogue", "对白与旁白"),
            new Field("shotType", "景别"),
            new Field("cameraMovement", "运镜"),
            new Field("cameraAngle", "拍摄角度"),
            new Field("cameraEquipment", "摄影设备"),
            new Field("focalLength", "焦距"),
            new Field("sound", "声音"),
            new Field("soundEffect", "音效"),
            new Field("music", "音乐"),
            new Field("transition", "转场"),
            new Field("remark", "备注"));
    private static final List<Field> ASSET_FIELDS = List.of(
            new Field("name", "名称"), new Field("description", "描述"), new Field("tags", "标签"));
    private static final Map<String, FieldDef> ASSET_PROPERTIES = assetProperties();
    private static final Set<String> EXCLUDED_PROPERTIES = Set.of(
            "id", "projectId", "userId", "teamId", "ownerId", "ownerType", "assetId", "sourceType",
            "itemType", "sortOrder", "status", "deleted", "version", "assetVersion", "createTime", "updateTime",
            "create_time", "update_time", "aiPrompt", "firstFramePrompt", "lastFramePrompt");

    private CanvasVideoPrompt() {
    }

    public static boolean appliesTo(CanvasNode node) {
        return node.kind() == CanvasNodeKind.VIDEO_GENERATION
                || node.kind() == CanvasNodeKind.COMFYUI && node.config().path("resultType").asText().equals("video");
    }

    public static boolean emptyPrompt(List<Value> values) {
        return values.isEmpty() || values.stream().anyMatch(value -> value.type() == CanvasValueType.TEXT
                && (value.text() == null || value.text().isBlank()));
    }

    public static String shot(ObjectNode data) {
        return String.join("\n", fields(data, SHOT_FIELDS));
    }

    public static String asset(ObjectNode data) {
        List<String> lines = fields(data, ASSET_FIELDS);
        data.path("properties").properties().forEach(entry -> appendProperty(lines, entry.getKey(), entry.getValue()));
        return String.join("\n", lines);
    }

    private static List<String> fields(ObjectNode data, List<Field> fields) {
        List<String> lines = new ArrayList<>();
        for (Field field : fields) {
            String text = render(data.path(field.key()));
            if (!text.isBlank()) {
                lines.add(field.label() + "：" + text);
            }
        }
        return lines;
    }

    private static void appendProperty(List<String> lines, String key, JsonNode value) {
        if (EXCLUDED_PROPERTIES.contains(key) || key.startsWith("_canvas")
                || key.endsWith("Url") || key.endsWith("Urls")) {
            return;
        }
        String text = render(value);
        if (text.isBlank()) {
            return;
        }
        String label = key;
        if (ASSET_PROPERTIES.containsKey(key)) {
            FieldDef field = ASSET_PROPERTIES.get(key);
            label = key.equals("description") ? "描述" : field.label();
            if (field.type().equals("select")) {
                for (var option : field.options()) {
                    if (option.value().equals(text)) {
                        text = option.label();
                        break;
                    }
                }
            }
        }
        lines.add(label + "：" + text);
    }

    private static String render(JsonNode value) {
        if (value.isMissingNode() || value.isNull()) {
            return "";
        }
        if (value.isObject()) {
            List<String> parts = new ArrayList<>();
            value.properties().forEach(entry -> appendProperty(parts, entry.getKey(), entry.getValue()));
            return String.join("；", parts);
        }
        if (value.isArray()) {
            List<String> parts = new ArrayList<>();
            for (JsonNode item : value) {
                String text = render(item);
                if (!text.isBlank()) {
                    parts.add(text);
                }
            }
            return String.join("；", parts);
        }
        return value.asText().strip();
    }

    private static Map<String, FieldDef> assetProperties() {
        Map<String, FieldDef> fields = new LinkedHashMap<>();
        for (String type : AssetMetadataRegistry.getSupportedTypes()) {
            for (FieldDef field : AssetMetadataRegistry.getFields(type)) {
                fields.put(field.key(), field);
            }
        }
        return Map.copyOf(fields);
    }
}
