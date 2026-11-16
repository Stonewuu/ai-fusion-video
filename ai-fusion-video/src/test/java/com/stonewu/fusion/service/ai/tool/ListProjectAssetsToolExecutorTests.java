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
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import com.stonewu.fusion.service.resource.model.ResourceSnapshot;
import com.stonewu.fusion.service.system.SystemConfigService;
import org.junit.jupiter.api.Test;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.ASSET_ITEM;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;

class ListProjectAssetsToolExecutorTests {
    private final ObjectMapper mapper = new ObjectMapper();
    private final CanvasActor actor = new CanvasActor(7, null);
    private final AssetResourceQueryService assets = mock(AssetResourceQueryService.class);
    private final ResourceQueryService resources = mock(ResourceQueryService.class);
    private final SystemConfigService config = mock(SystemConfigService.class);
    private final AssetToolQueries tools = new AssetToolQueries(assets, resources, config, new CanvasJson(mapper));
    private final ToolExecutionContext invocation = ToolExecutionContext.forInvocation(7L, 1, 7L,
            new ResourceCommandContext(actor, "agent", null, "run", "call", "owner", 1L));

    @Test
    void listsFilteredSummaryPageAndKeepsTotalIndependentOfCurrentPage() throws Exception {
        var page = mapper.createObjectNode().put("total", 176).put("page", 2).put("pageSize", 1)
                .put("hasMore", true).put("nextPage", 3);
        page.putArray("assets").add(mapper.createObjectNode().put("id", 19).put("version", 3)
                .put("name", "龙").put("itemCount", 12).put("descriptionPreview", "青色巨龙").put("hasImage", 1));
        when(assets.summaryPage(1L, "character", "龙", 2, 1, actor)).thenReturn(page);
        var result = mapper.readTree(new ListProjectAssetsToolExecutor(tools).execute(
                "{\"projectId\":1,\"type\":\"character\",\"keyword\":\"龙\",\"page\":2,\"pageSize\":1}", invocation));
        assertThat(result.path("total").intValue()).isEqualTo(176);
        assertThat(result.path("nextPage").intValue()).isEqualTo(3);
        assertThat(result.path("assets").get(0).path("assetId").intValue()).isEqualTo(19);
        assertThat(result.path("assets").get(0).path("hasImage").booleanValue()).isTrue();
        assertThat(result.path("detailsTool").textValue()).isEqualTo("get_asset");
        verifyNoInteractions(resources, config);
    }

    @Test
    void readsOnlyChosenItemWithResolvedUrlsAndCurrentVersions() throws Exception {
        var item = mapper.createObjectNode().put("id", 21).put("assetId", 19).put("version", 2).put("assetVersion", 3)
                .put("imageUrl", "/media/images/dragon.png")
                .put("thumbnailUrl", "/media/images/private-thumb.png");
        item.set("properties", mapper.createObjectNode().put("description", "完整外观".repeat(1000)));
        when(resources.get(ASSET_ITEM, 21, actor)).thenReturn(new ResourceSnapshot(
                new ResourceHeader(ASSET_ITEM, 21, 1L, 2, false), item));
        when(config.resolvePublicUrl("/media/images/dragon.png")).thenReturn("https://example.com/media/images/dragon.png");
        var result = mapper.readTree(tools.execute("get_asset_item", "{\"assetItemId\":21}", invocation));
        assertThat(result.path("properties").path("description").textValue()).hasSize(4000);
        assertThat(result.path("imageUrl").textValue()).isEqualTo("https://example.com/media/images/dragon.png");
        assertThat(result.path("thumbnailUrl").isNull()).isTrue();
        assertThat(result.path("thumbnailUrlStored").textValue()).isEqualTo("/media/images/private-thumb.png");
        assertThat(result.path("assetItemId").intValue()).isEqualTo(21);
        assertThat(result.path("assetVersion").intValue()).isEqualTo(3);
        verifyNoInteractions(assets);
    }

    @Test
    void rejectsAmbiguousLookupAndInvalidPageBeforeQuerying() throws Exception {
        for (String input : new String[]{"{\"assetId\":1,\"assetName\":\"龙\"}", "{\"assetIds\":[1,1]}",
                "{\"assetName\":\"龙\"}", "{\"assetId\":1,\"page\":1.5}"}) {
            assertThat(mapper.readTree(tools.execute("query_asset_items", input, invocation)).path("status").textValue())
                    .isEqualTo("error");
        }
        verifyNoInteractions(assets, resources);
    }
}
