package com.stonewu.fusion.service.ai.catalog;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.doReturn;
import static org.mockito.ArgumentMatchers.any;

class ProviderCatalogServiceTests {
    private final ProviderCatalogService catalog = new ProviderCatalogService(new ObjectMapper());

    ProviderCatalogServiceTests() throws Exception {
    }

    /** 终止赞助只影响展示，不能放宽密钥要求或关闭提供商的快速配置能力。 */
    @Test
    void removingSponsorshipKeepsProviderConfigurationContract() throws Exception {
        var json = new ObjectMapper();
        Map<String, Object> definition = json.convertValue(catalog.require("metaso"), new TypeReference<>() {});
        definition.put("sponsored", false);
        var independent = json.convertValue(definition, ProviderDefinition.class);
        var source = mock(ObjectMapper.class);
        doReturn(List.of(independent)).when(source).readValue(any(InputStream.class), any(TypeReference.class));
        var updated = new ProviderCatalogService(source);
        assertThat(updated.require("metaso").sponsored()).isFalse();
        assertThat(updated.require("metaso").quickSetup()).isTrue();
        var config = ApiConfig.builder().providerId("metaso").platform("openai_compatible").build();
        assertThatThrownBy(() -> updated.validateBinding(config)).hasMessageContaining("API Key");
        config.setStatus(0);
        assertThatCode(() -> updated.validateBinding(config)).doesNotThrowAnyException();
        config.setStatus(1);
        assertThatThrownBy(() -> updated.validateBinding(config)).hasMessageContaining("API Key");
        config.setApiKey("fixture-key");
        assertThatCode(() -> updated.validateBinding(config)).doesNotThrowAnyException();
    }

    /** 自定义接入无需平台身份，但显式绑定的平台必须存在且鉴权类型一致。 */
    @Test
    void validatesExplicitIdentityWithoutClaimingCustomConfigurations() {
        var custom = ApiConfig.builder().platform("anthropic").apiKey("local-test-key").build();
        assertThatCode(() -> catalog.validateBinding(custom)).doesNotThrowAnyException();
        custom.setProviderId("metaso");
        assertThatThrownBy(() -> catalog.validateBinding(custom))
                .isInstanceOf(BusinessException.class).hasMessageContaining("接入类型不匹配");
        custom.setPlatform("openai_compatible");
        assertThatCode(() -> catalog.validateBinding(custom)).doesNotThrowAnyException();
        custom.setProviderId("unregistered-provider");
        assertThatThrownBy(() -> catalog.validateBinding(custom)).hasMessageContaining("提供商不存在");
    }

    /** 相同模型经不同渠道调用时，赞助身份不得绕过显式协议选择。 */
    @Test
    void modelProtocolOverridesAreIndependentOfSponsorship() {
        var resolver = new AiModelMetadataResolver(mock(ApiConfigService.class));
        var config = ApiConfig.builder().providerId("metaso").platform("openai_compatible")
                .videoProtocol("metaso_video").build();
        var model = AiModel.builder().code("MiniMax-H3").modelType(3).build();
        assertThat(resolver.resolve(model, config).modelProtocol()).isEqualTo("metaso_video");
        model.setModelProtocol("minimax_h3");
        assertThat(resolver.resolve(model, config).modelProtocol()).isEqualTo("minimax_h3");
        config.setProviderId(null);
        assertThat(resolver.resolve(model, config).modelProtocol()).isEqualTo("minimax_h3");
    }
}
