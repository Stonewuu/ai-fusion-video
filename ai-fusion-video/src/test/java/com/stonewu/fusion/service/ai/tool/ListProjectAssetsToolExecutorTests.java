package com.stonewu.fusion.service.ai.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.asset.AssetToolQueries;
import com.stonewu.fusion.service.ai.tool.asset.ListProjectAssetsToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.resource.AssetResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.system.SystemConfigService;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class ListProjectAssetsToolExecutorTests {
    @Test
    void returnsResolvedUrlsAndVersionsWithoutPretendingUnresolvedPathsArePublic() {
        var assets = mock(AssetResourceQueryService.class);
        var config = mock(SystemConfigService.class);
        var mapper = new ObjectMapper();
        var actor = new CanvasActor(7, null);
        var tools = new AssetToolQueries(assets, mock(ResourceQueryService.class), config, new CanvasJson(mapper));
        var executor = new ListProjectAssetsToolExecutor(tools);
        var asset = mapper.createObjectNode().put("id", 19).put("version", 3).put("name", "dragon").put("coverUrl", "/media/images/dragon.png");
        asset.putArray("items").add(mapper.createObjectNode().put("id", 21).put("assetId", 19).put("version", 2).put("assetVersion", 3)
                .put("imageUrl", "/media/images/dragon.png").put("thumbnailUrl", "/media/images/private-thumb.png"));
        when(assets.list(1L, "character", null, true, actor)).thenReturn(List.of(asset));
        when(config.resolvePublicUrl("/media/images/dragon.png")).thenReturn("https://example.com/media/images/dragon.png");
        var invocation = ToolExecutionContext.forInvocation(7L, 1, 7L,
                new ResourceCommandContext(actor, "agent", null, "run", "call", "owner", 1L));
        String result = executor.execute("{\"projectId\":1,\"type\":\"character\"}", invocation);
        assertThat(result).contains("\"imageUrl\":\"https://example.com/media/images/dragon.png\"", "\"assetVersion\":3",
                "\"thumbnailUrl\":null", "\"thumbnailUrlStored\":\"/media/images/private-thumb.png\"");
        verify(assets).list(1L, "character", null, true, actor);
    }
}
