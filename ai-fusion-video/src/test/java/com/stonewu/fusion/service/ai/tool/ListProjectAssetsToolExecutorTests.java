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

import java.util.List;
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
    void readsBatchItemDirectoriesOnceAndKeepsEachAssetsPageAndVersion() throws Exception {
        var first = mapper.createObjectNode().put("id", 19).put("version", 4).put("name", "青龙")
                .put("itemCount", 3).put("page", 2).put("pageSize", 1).put("hasMore", true).put("nextPage", 3);
        first.putArray("items").add(mapper.createObjectNode().put("id", 21).put("assetId", 19)
                .put("version", 2).put("assetVersion", 4).put("name", "夜间变体")
                .put("descriptionPreview", "深青色鳞片").put("hasImage", 1));
        var second = mapper.createObjectNode().put("id", 23).put("version", 8).put("name", "山门")
                .put("itemCount", 1).put("page", 2).put("pageSize", 1).put("hasMore", false);
        second.putNull("nextPage");
        second.putArray("items");
        when(assets.itemSummaryPages(List.of(19L, 23L), 2, 1, actor)).thenReturn(List.of(first, second));

        var result = mapper.readTree(tools.execute("query_asset_items",
                "{\"assetIds\":[19,23],\"page\":2,\"pageSize\":1}", invocation));

        assertThat(result.path("status").asText()).isEqualTo("success");
        assertThat(result.path("total").asInt()).isEqualTo(2);
        assertThat(result.path("detailsTool").asText()).isEqualTo("get_asset_item");
        var rows = result.path("assets");
        assertThat(rows.get(0).path("assetId").asLong()).isEqualTo(19);
        assertThat(rows.get(0).path("nextPage").asInt()).isEqualTo(3);
        var item = rows.get(0).path("items").get(0);
        assertThat(item.path("assetItemId").asLong()).isEqualTo(21);
        assertThat(item.path("assetId").asLong()).isEqualTo(19);
        assertThat(item.path("version").asInt()).isEqualTo(2);
        assertThat(item.path("assetVersion").asInt()).isEqualTo(4);
        assertThat(item.path("hasImage").asBoolean()).isTrue();
        assertThat(item.path("descriptionPreview").asText()).isEqualTo("深青色鳞片");
        assertThat(rows.get(1).path("assetId").asLong()).isEqualTo(23);
        assertThat(rows.get(1).path("items").isEmpty()).isTrue();
        assertThat(rows.get(1).path("hasMore").asBoolean()).isFalse();
        assertThat(rows.get(1).path("nextPage").isNull()).isTrue();
        verify(assets).itemSummaryPages(List.of(19L, 23L), 2, 1, actor);
        verifyNoMoreInteractions(assets);
        verifyNoInteractions(resources, config);
    }

    @Test
    void singleAssetDirectoryUsesTheSameSummaryBatchContract() throws Exception {
        var row = mapper.createObjectNode().put("id", 19).put("version", 5).put("itemCount", 0)
                .put("page", 1).put("pageSize", 20).put("hasMore", false);
        row.putNull("nextPage");
        row.putArray("items");
        when(assets.itemSummaryPages(List.of(19L), 1, 20, actor)).thenReturn(List.of(row));

        var result = mapper.readTree(tools.execute("query_asset_items", "{\"assetId\":19}", invocation));

        assertThat(result.path("status").asText()).isEqualTo("success");
        assertThat(result.path("assets").get(0).path("assetId").asLong()).isEqualTo(19);
        assertThat(result.path("assets").get(0).path("items").isEmpty()).isTrue();
        assertThat(result.path("assets").get(0).path("version").asInt()).isEqualTo(5);
        verify(assets).itemSummaryPages(List.of(19L), 1, 20, actor);
        verifyNoMoreInteractions(assets);
        verifyNoInteractions(resources, config);
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
