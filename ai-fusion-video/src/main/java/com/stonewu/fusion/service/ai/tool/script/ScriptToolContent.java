package com.stonewu.fusion.service.ai.tool.script;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/** 写入回执可能嵌套分集快照；移除全部层级的原文，保留版本和业务结果。 */
final class ScriptToolContent {
    private ScriptToolContent() {
    }

    static void omitSource(JsonNode value) {
        if (value instanceof ObjectNode object) {
            object.remove("rawContent");
            if (!object.has("type")) {
                object.remove("content");
            }
        }
        if (value.isContainerNode()) {
            value.forEach(ScriptToolContent::omitSource);
        }
    }
}
