package com.stonewu.fusion.service.generation.video.strategy;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class VideoGenerationStrategyRouterTests {

    @Test
    void routesByProviderDefaultAndSupportsExplicitNewApiSubprotocols() {
        ApiConfigService apiConfigService = mock(ApiConfigService.class);
        AiModelMetadataResolver resolver = new AiModelMetadataResolver(apiConfigService);

        VideoGenerationStrategy newApiStrategy = mock(VideoGenerationStrategy.class);
        when(newApiStrategy.getSupportedProtocols()).thenReturn(Set.of("newapi", "jimeng", "kling", "sora"));
        VideoGenerationStrategy agnesStrategy = mock(VideoGenerationStrategy.class);
        when(agnesStrategy.getSupportedProtocols()).thenReturn(Set.of("agnes"));

        VideoGenerationStrategyRouter router = new VideoGenerationStrategyRouter(
                List.of(newApiStrategy, agnesStrategy), resolver);

        AiModel inherited = AiModel.builder().id(1L).apiConfigId(11L).modelType(3).build();
        when(apiConfigService.getById(11L)).thenReturn(ApiConfig.builder()
                .id(11L)
                .platform("openai_compatible")
                .videoProtocol("agnes")
                .build());
        assertSame(agnesStrategy, router.resolve(inherited));

        AiModel overridden = AiModel.builder()
                .id(2L)
                .apiConfigId(12L)
                .modelType(3)
                .modelProtocol("kling")
                .build();
        when(apiConfigService.getById(12L)).thenReturn(ApiConfig.builder()
                .id(12L)
                .platform("newapi")
                .videoProtocol("newapi")
                .build());
        assertSame(newApiStrategy, router.resolve(overridden));
    }

    @Test
    void rejectsMissingVideoProtocolInsteadOfRoutingByPlatform() {
        ApiConfigService apiConfigService = mock(ApiConfigService.class);
        VideoGenerationStrategy newApiStrategy = mock(VideoGenerationStrategy.class);
        when(newApiStrategy.getSupportedProtocols()).thenReturn(Set.of("newapi"));
        VideoGenerationStrategyRouter router = new VideoGenerationStrategyRouter(
                List.of(newApiStrategy), new AiModelMetadataResolver(apiConfigService));
        AiModel model = AiModel.builder().id(1L).apiConfigId(11L).modelType(3).build();
        when(apiConfigService.getById(11L)).thenReturn(
                ApiConfig.builder().id(11L).platform("newapi").build());

        assertThatThrownBy(() -> router.resolve(model))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("未配置请求协议");
    }
}
