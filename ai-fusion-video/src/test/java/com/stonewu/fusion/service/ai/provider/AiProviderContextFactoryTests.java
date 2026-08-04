package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AiProviderContextFactoryTests {

    @Test
    void createForApiConfigNormalizesLegacyOpenAiPlatform() {
        AiProviderContextFactory factory = new AiProviderContextFactory(null, mock(AiModelMetadataResolver.class));

        AiProviderContext context = factory.createForApiConfig(
                ApiConfig.builder().platform("openai").apiUrl("https://api.openai.com").build());

        assertThat(context.getPlatform()).isEqualTo("openai_compatible");
        // 默认地址在库中可能为 null，创建上下文时应还原为平台默认根地址
        assertThat(context.getBaseUrl()).isEqualTo("https://api.openai.com");
    }

    @Test
    void createForApiConfigRestoresAgnesDefaultUrlWhenApiUrlNull() {
        AiProviderContextFactory factory = new AiProviderContextFactory(null, mock(AiModelMetadataResolver.class));

        AiProviderContext context = factory.createForApiConfig(
                ApiConfig.builder()
                        .platform("agnes")
                        .apiKey("agnes-key")
                        .apiUrl(null)
                        .textProtocol("openai_compatible")
                        .build());

        assertThat(context.getPlatform()).isEqualTo("agnes");
        assertThat(context.getBaseUrl()).isEqualTo("https://apihub.agnes-ai.com");
        assertThat(context.getApiKey()).isEqualTo("agnes-key");
    }

    @Test
    void createForModelUsesTextProtocolButRestoresAuthPlatformDefaultUrl() {
        ApiConfigService apiConfigService = mock(ApiConfigService.class);
        AiModelMetadataResolver metadataResolver = mock(AiModelMetadataResolver.class);
        AiProviderContextFactory factory = new AiProviderContextFactory(apiConfigService, metadataResolver);

        ApiConfig apiConfig = ApiConfig.builder()
                .id(11L)
                .platform("agnes")
                .apiKey("agnes-key")
                .apiUrl(null)
                .textProtocol("openai_compatible")
                .build();
        AiModel model = AiModel.builder()
                .id(22L)
                .apiConfigId(11L)
                .modelType(1)
                .code("agnes-2.0-flash")
                .name("Agnes 2.0 Flash")
                .build();

        when(apiConfigService.getById(11L)).thenReturn(apiConfig);
        when(apiConfigService.resolveEffectiveApiUrl(apiConfig)).thenReturn("https://apihub.agnes-ai.com");
        when(metadataResolver.resolve(eq(model), eq(apiConfig)))
                .thenReturn(new AiModelMetadata("agnes", "agnes", "openai_compatible"));

        AiProviderContext context = factory.createForModel(model);

        // 文本请求协议用于路由
        assertThat(context.getPlatform()).isEqualTo("openai_compatible");
        // 但 baseUrl 必须按鉴权平台 Agnes 还原，不能落到 OpenAI
        assertThat(context.getBaseUrl()).isEqualTo("https://apihub.agnes-ai.com");
        assertThat(context.getApiKey()).isEqualTo("agnes-key");
        assertThat(context.getModelName()).isEqualTo("agnes-2.0-flash");
    }

    @Test
    void createForApiConfigRequiresExplicitAccessType() {
        AiProviderContextFactory factory = new AiProviderContextFactory(null, mock(AiModelMetadataResolver.class));

        assertThatThrownBy(() -> factory.createForApiConfig(
                ApiConfig.builder().apiUrl("https://api.openai.com").build()))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("接入与鉴权类型");
    }
}
