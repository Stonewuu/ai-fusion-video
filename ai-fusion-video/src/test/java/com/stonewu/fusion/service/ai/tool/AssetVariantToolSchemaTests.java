package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.tool.asset.AssetToolCommands;
import com.stonewu.fusion.service.ai.tool.asset.BatchCreateAssetItemsToolExecutor;
import com.stonewu.fusion.service.resource.AssetFields;
import com.stonewu.fusion.service.canvas.CanvasJson;
import org.junit.jupiter.api.Test;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET_ITEM;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.Mockito.mock;

class AssetVariantToolSchemaTests {
    /** 工具声明的子资产字段必须能通过实际保存校验，避免再次引导模型提交顶层 description。 */
    @Test
    void advertisedVariantFieldsAreAcceptedByTheResourceWriter() throws Exception {
        var json = new ObjectMapper();
        var tool = new BatchCreateAssetItemsToolExecutor(mock(AssetToolCommands.class));
        var properties = json.readTree(tool.getParametersSchema()).at("/properties/items/items/properties");
        var input = json.createObjectNode();
        properties.fields().forEachRemaining(field -> {
            if (field.getValue().path("type").asText().equals("object")) {
                input.putObject(field.getKey()).put("description", "保持人物身份，仅更换服装");
            } else {
                input.put(field.getKey(), "variant");
            }
        });
        assertThatCode(() -> new AssetFields(new CanvasJson(json)).values(ASSET_ITEM, input, true))
                .doesNotThrowAnyException();
    }
}
