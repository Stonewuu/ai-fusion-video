package com.stonewu.fusion.service.ai.tool;

import com.stonewu.fusion.service.ai.tool.asset.ListProjectAssetsToolExecutor;
import com.stonewu.fusion.entity.asset.Asset;
import com.stonewu.fusion.entity.asset.AssetItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.asset.AssetService;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.system.SystemConfigService;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ListProjectAssetsToolExecutorTests {

    @Test
    void returnsAbsolutePublicUrlsForAssetImages() {
        AssetService assetService = mock(AssetService.class);
        ProjectService projectService = mock(ProjectService.class);
        SystemConfigService systemConfigService = mock(SystemConfigService.class);
        ListProjectAssetsToolExecutor executor = new ListProjectAssetsToolExecutor(
                assetService, projectService, systemConfigService);

        Asset asset = Asset.builder()
                .id(19L)
                .projectId(1L)
                .type("character")
                .name("dragon")
                .coverUrl("/media/images/dragon.png")
                .build();
        AssetItem item = AssetItem.builder()
                .id(21L)
                .assetId(19L)
                .name("front")
                .imageUrl("/media/images/dragon.png")
                .thumbnailUrl("https://cdn.example.com/dragon-thumb.png")
                .build();

        when(projectService.canAccessProject(1L, 7L)).thenReturn(true);
        when(assetService.listByProject(1L, "character", null)).thenReturn(List.of(asset));
        when(assetService.listItems(19L)).thenReturn(List.of(item));
        when(systemConfigService.resolvePublicUrl("/media/images/dragon.png"))
                .thenReturn("http://localhost:3000/media/images/dragon.png");
        when(systemConfigService.resolvePublicUrl("https://cdn.example.com/dragon-thumb.png"))
                .thenReturn("https://cdn.example.com/dragon-thumb.png");

        String result = executor.execute(
                "{\"projectId\":1,\"type\":\"character\"}",
                ToolExecutionContext.builder().userId(7L).build());

        assertThat(result)
                .contains("\"coverUrl\":\"http://localhost:3000/media/images/dragon.png\"")
                .contains("\"imageUrl\":\"http://localhost:3000/media/images/dragon.png\"")
                .contains("\"thumbnailUrl\":\"https://cdn.example.com/dragon-thumb.png\"")
                .doesNotContain("\"coverUrl\":\"/media/images/dragon.png\"")
                .doesNotContain("\"imageUrl\":\"/media/images/dragon.png\"");
    }
}
