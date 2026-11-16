package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.asset.AssetToolCommands;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AssetToolResultProjectionTests {
    @Test
    void writeResultKeepsIdentityAndUnicodeSummaryWithoutChangingTheDurableReceipt() throws Exception {
        var mapper = new ObjectMapper();
        var commands = mock(ResourceCommandService.class);
        String description = "星🌟".repeat(150);
        var durable = mapper.createObjectNode().put("id", 17).put("version", 4).put("name", "测试资产")
                .put("description", description).put("aiPrompt", "完整提示词");
        durable.putObject("properties").put("appearance", "完整外观");
        var item = durable.putArray("items").addObject().put("id", 18).put("assetId", 17)
                .put("version", 2).put("assetVersion", 4).put("name", "夜间变体").put("aiPrompt", "子资产提示词");
        item.putObject("properties").put("description", "子资产完整描述");
        var original = durable.deepCopy();
        when(commands.execute(any(), any())).thenReturn(new ResourceCommandReceipt("command", "request", durable, List.of()));
        var tools = new AssetToolCommands(commands, mock(ResourceQueryService.class), new CanvasJson(mapper));
        var actor = new CanvasActor(7, null);
        var invocation = ToolExecutionContext.forInvocation(7L, 1, 7L,
                new ResourceCommandContext(actor, "agent", null, "run", "call", "owner", 1L));

        var result = mapper.readTree(tools.execute("create_asset", "{\"type\":\"prop\",\"name\":\"测试资产\"}", invocation));

        assertThat(result.path("status").asText()).isEqualTo("success");
        assertThat(result.path("descriptionPreview").asText()).isEqualTo("星🌟".repeat(100));
        assertThat(result.has("description") || result.has("properties") || result.has("aiPrompt")).isFalse();
        var child = result.path("items").get(0);
        assertThat(child.path("assetItemId").asInt()).isEqualTo(18);
        assertThat(child.path("assetVersion").asInt()).isEqualTo(4);
        assertThat(child.path("version").asInt()).isEqualTo(2);
        assertThat(child.has("properties") || child.has("aiPrompt")).isFalse();
        assertThat(durable).isEqualTo(original);
    }
}
