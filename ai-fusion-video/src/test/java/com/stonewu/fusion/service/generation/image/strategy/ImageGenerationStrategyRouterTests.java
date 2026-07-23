package com.stonewu.fusion.service.generation.image.strategy;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ImageGenerationStrategyRouterTests {

    @Test
    void routesByModelOverrideOrProviderImageDefaultOnly() {
        ImageGenerationStrategy openAi = mock(ImageGenerationStrategy.class);
        when(openAi.getName()).thenReturn("openai");
        ImageGenerationStrategy agnes = mock(ImageGenerationStrategy.class);
        when(agnes.getName()).thenReturn("agnes");

        ImageGenerationStrategyRouter router = new ImageGenerationStrategyRouter(
                List.of(openAi, agnes),
                new AiModelMetadataResolver(mock(ApiConfigService.class))
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai_compatible")
                .imageProtocol("agnes")
                .build();

        assertSame(agnes, router.resolve(AiModel.builder().modelType(2).build(), apiConfig));
        assertSame(openAi, router.resolve(AiModel.builder()
                .modelType(2)
                .modelProtocol("openai")
                .build(), apiConfig));
    }

    @Test
    void rejectsMissingImageProtocolInsteadOfGuessingFromModelCodeOrPlatform() {
        ImageGenerationStrategy openAi = mock(ImageGenerationStrategy.class);
        when(openAi.getName()).thenReturn("openai");
        ImageGenerationStrategyRouter router = new ImageGenerationStrategyRouter(
                List.of(openAi),
                new AiModelMetadataResolver(mock(ApiConfigService.class))
        );

        assertThatThrownBy(() -> router.resolve(
                AiModel.builder().modelType(2).code("agnes-image-2.1-flash").build(),
                ApiConfig.builder().platform("openai_compatible").build()))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("未配置请求协议");
    }
}
