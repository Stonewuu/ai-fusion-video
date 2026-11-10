package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** 保护目录中的独立版本和能力分类，仅保留四类生成模型及各自的调用协议。 */
class GeminiCatalogClassificationTests {
    @Test
    void preservesVideoProtocolWhenEnrichingTheRemoteCatalog() {
        var contextFactory = mock(AiProviderContextFactory.class);
        var registry = mock(AiProviderRegistry.class);
        var provider = mock(AiProvider.class);
        var context = AiProviderContext.builder().platform("vertex_ai").build();
        when(contextFactory.createForApiConfig(8L)).thenReturn(context);
        when(registry.getProvider(context)).thenReturn(provider);
        when(provider.listRemoteModels(context)).thenReturn(List.of(RemoteModelVO.builder()
                .id("gemini-omni-1.2-flash").modelType(3).modelProtocol("vertex_ai").build()));
        var service = new AiProviderService(contextFactory, registry,
                new AiModelMetadataResolver(mock(ApiConfigService.class)), mock(ModelPresetService.class));

        assertThat(service.listRemoteModels(8L)).singleElement().satisfies(model -> {
            assertThat(model.getModelType()).isEqualTo(3);
            assertThat(model.getModelProtocol()).isEqualTo("vertex_ai");
        });
    }

    @Test
    void preservesFullModelIdsAndClassifiesNativeGenerationCapabilities() {
        var provider = new GeminiAiProvider() {
            @Override
            protected String executeGet(String url, Map<String, String> headers, ApiConfig api) {
                return """
                        {"models":[
                          {"name":"models/gemini-3.8-flash","supportedGenerationMethods":["generateContent"]},
                          {"name":"models/gemini-3.1-flash-image","baseModelId":"gemini-3.1-flash-image","supportedGenerationMethods":["generateContent"]},
                          {"name":"models/gemini-3.1-flash-image-preview","baseModelId":"gemini-3.1-flash-image","supportedGenerationMethods":["generateContent"]},
                          {"name":"models/gemini-3.8-flash-tts","supportedGenerationMethods":["generateContent"]},
                          {"name":"models/gemini-embedding-2","supportedGenerationMethods":["embedContent"]},
                          {"name":"models/gemini-3.5-transcribe","supportedGenerationMethods":["generateContent"]},
                          {"name":"models/gemini-3.8-live","supportedGenerationMethods":["generateContent"]},
                          {"name":"models/deep-research-preview-04-2026","supportedGenerationMethods":["generateContent"]},
                          {"name":"models/lyria-3-clip-preview","supportedGenerationMethods":[]},
                          {"name":"models/gemini-omni-1.1-flash","supportedGenerationMethods":["generateContent"]},
                          {"name":"models/gemini-omni-1.2-flash","supportedGenerationMethods":[]}
                        ]}
                        """;
            }
        };
        var results = provider.listRemoteModels(AiProviderContext.builder().apiKey("fixture-key")
                .apiConfig(ApiConfig.builder().platform("gemini").build()).build());
        assertThat(results).extracting(RemoteModelVO::getId).containsExactly(
                "gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-3.8-flash",
                "gemini-3.8-flash-tts", "gemini-omni-1.1-flash", "gemini-omni-1.2-flash", "lyria-3-clip-preview");
        assertThat(results).extracting(RemoteModelVO::getModelType).containsExactly(2, 2, 1, 4, 3, 3, 4);
        assertThat(results).allSatisfy(model -> assertThat(model.getModelProtocol()).isEqualTo("gemini"));
    }
}
