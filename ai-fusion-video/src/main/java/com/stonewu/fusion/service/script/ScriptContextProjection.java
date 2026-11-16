package com.stonewu.fusion.service.script;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

/** 画布上下文只携带剧本摘要；单集原文通过专用读取工具获取。 */
public final class ScriptContextProjection {
    private ScriptContextProjection() {
    }

    public static ObjectNode summary(CanvasValueType type, ObjectNode data) {
        ObjectNode result = data.deepCopy();
        if (type == CanvasValueType.SCRIPT || type == CanvasValueType.SCRIPT_EPISODE) {
            result.remove("rawContent");
            result.remove("content");
        }
        return result;
    }

    public static void omitSource(JsonNode value) {
        if (value instanceof ObjectNode object) {
            if (object.has("totalEpisodes") && object.has("projectId")) {
                object.remove("content");
            }
            object.remove("rawContent");
        }
        if (value.isContainerNode()) {
            value.forEach(ScriptContextProjection::omitSource);
        }
    }
}
