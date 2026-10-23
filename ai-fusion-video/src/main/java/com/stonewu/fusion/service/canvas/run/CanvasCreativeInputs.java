package com.stonewu.fusion.service.canvas.run;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Compose explicitly connected textual context. Visual media must use a separate media input.
 */
public final class CanvasCreativeInputs {
    private static final Set<String> TECHNICAL_FIELDS = Set.of("id", "projectId", "userId", "teamId", "assetId",
            "scriptId",
            "episodeId", "sceneId", "storyboardId", "version", "deleted", "createTime", "updateTime", "create_time",
            "update_time");

    private CanvasCreativeInputs() {
    }

    public static void compose(CanvasNode node, Map<String, List<Value>> inputs, CanvasRunSources sources) {
        List<Value> context = inputs.getOrDefault("context", List.of());
        if (context.isEmpty()) {
            return;
        }
        List<Value> prompts = inputs.getOrDefault("prompt", List.of());
        if (prompts.size() > 1 || prompts.stream().anyMatch(value -> value.type() != CanvasValueType.TEXT)) {
            throw CanvasRunException.invalid("创作要求只能提供一份文本；多份内容请连接到内容设定");
        }
        List<String> sections = new ArrayList<>();
        if (!prompts.isEmpty() && prompts.getFirst().text() != null && !prompts.getFirst().text().isBlank()) {
            sections.add("创作要求：\n" + prompts.getFirst().text());
        }
        for (int index = 0; index < context.size(); index++) {
            Value value = context.get(index);
            String content;
            if (value.type() == CanvasValueType.TEXT) {
                content = value.text();
            } else if (value.type().isBusinessResource() && value.resource() != null && value.data() != null) {
                ObjectNode data = value.data().deepCopy();
                removeMediaAndMetadata(data);
                content = data.isEmpty() ? "" : data.toPrettyString();
            } else {
                throw CanvasRunException.invalid("内容设定只接受文字、资产或项目内容；图片和视频请接入相应媒体输入");
            }
            if (content == null || content.isBlank()) {
                throw CanvasRunException.invalid(
                        "第 " + (index + 1) + " 份内容设定没有可用文本，请补充内容或选择媒体输入");
            }
            sections.add("内容设定 " + (index + 1) + "：\n" + content);
        }
        String prompt = String.join("\n\n", sections);
        int limit = node.kind() == CanvasNodeKind.TEXT_GENERATION ? 64000 : 32000;
        if (prompt.length() > limit) {
            throw CanvasRunException.invalid("组合后的创作内容超过 " + limit + " 字符，请减少内容设定或拆分任务");
        }
        inputs.put("prompt", List.of(sources.text(prompt)));
    }

    private static void removeMediaAndMetadata(ObjectNode object) {
        List<String> remove = new ArrayList<>();
        object.properties().forEach(entry -> {
            String key = entry.getKey();
            JsonNode value = entry.getValue();
            if (TECHNICAL_FIELDS.contains(key) || key.startsWith("_canvas") || key.endsWith("Url") || key.endsWith(
                    "Urls")) {
                remove.add(key);
            } else if (value.isObject()) {
                removeMediaAndMetadata((ObjectNode) value);
            } else if (value.isArray()) {
                value.forEach(item -> {
                    if (item.isObject()) {
                        removeMediaAndMetadata((ObjectNode) item);
                    }
                });
            }
        });
        object.remove(remove);
    }
}
