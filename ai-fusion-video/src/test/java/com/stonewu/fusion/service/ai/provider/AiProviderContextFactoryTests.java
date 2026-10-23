package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AiProviderContextFactoryTests {

    /** 导入未含密钥的连接保持停用，文本调用必须在构造上游客户端之前被拒绝。 */
    @Test
    void disabledConfigurationCannotProduceTextRuntimeUntilEnabled() {
        var configs = mock(ApiConfigService.class);
        var api = ApiConfig.builder().id(4L).providerId("infistar").platform("openai_compatible")
                .textProtocol("openai_compatible").apiKey("fixture-key").status(0).build();
        when(configs.getById(4L)).thenReturn(api);
        var factory = new AiProviderContextFactory(configs, new AiModelMetadataResolver(configs));
        var model = AiModel.builder().id(1L).apiConfigId(4L).modelType(1).code("text-model").build();
        assertThatThrownBy(() -> factory.createForModel(model)).isInstanceOf(BusinessException.class)
                .hasMessageContaining("已停用");
        api.setStatus(1);
        assertThat(factory.createForModel(model).getApiKey()).isEqualTo("fixture-key");
    }

    @Test
    void createForApiConfigNormalizesLegacyOpenAiPlatform() {
        AiProviderContextFactory factory = new AiProviderContextFactory(null, mock(AiModelMetadataResolver.class));

        AiProviderContext context = factory.createForApiConfig(
                ApiConfig.builder().platform("openai").apiUrl("https://api.openai.com").build());

        assertThat(context.getPlatform()).isEqualTo("openai_compatible");
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
