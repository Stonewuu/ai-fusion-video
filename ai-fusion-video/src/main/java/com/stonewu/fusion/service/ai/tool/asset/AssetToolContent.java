package com.stonewu.fusion.service.ai.tool.asset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

/** 写入回执保留识别对象、核对结果和继续更新所需的信息，完整设定按 ID 读取。 */
final class AssetToolContent {
    private AssetToolContent() {
    }

    static void summarize(ObjectNode result) {
        JsonNode description = result.remove("description");
        if (description != null && description.isTextual()) {
            String value = description.textValue();
            int length = Math.min(value.codePointCount(0, value.length()), 200);
            result.put("descriptionPreview", value.substring(0, value.offsetByCodePoints(0, length)));
        }
        result.remove(List.of("properties", "aiPrompt"));
        for (String field : List.of("items", "created", "existing", "allAssets", "allItems")) {
            for (JsonNode child : result.path(field)) {
                summarize((ObjectNode) child);
            }
        }
    }
}
