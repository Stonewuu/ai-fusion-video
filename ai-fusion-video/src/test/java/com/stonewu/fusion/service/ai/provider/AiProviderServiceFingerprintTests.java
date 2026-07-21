package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AiProviderServiceFingerprintTests {

    @Test
    void providerCredentialEndpointAndProxyChangesAlterKernelFingerprint() {
        AiProviderContextFactory contextFactory = mock(AiProviderContextFactory.class);
        AiProviderService service = new AiProviderService(
                contextFactory,
                mock(AiProviderRegistry.class),
                mock(AiModelMetadataResolver.class));
        AiModel model = AiModel.builder()
                .id(7L)
                .code("gpt-test")
                .apiConfigId(3L)
                .config("{\"temperature\":0.2}")
                .build();
        ApiConfig firstConfig = ApiConfig.builder()
                .id(3L)
                .platform("openai_compatible")
                .apiUrl("https://first.example/v1")
                .apiKey("first-key")
                .proxyType("http")
                .proxyHost("proxy-a")
                .proxyPort(8080)
                .build();
        ApiConfig secondConfig = ApiConfig.builder()
                .id(3L)
                .platform("openai_compatible")
                .apiUrl("https://second.example/v1")
                .apiKey("second-key")
                .proxyType("socks5")
                .proxyHost("proxy-b")
                .proxyPort(1080)
                .build();
        when(contextFactory.createForModel(model)).thenReturn(
                context(model, firstConfig),
                context(model, secondConfig));

        String first = service.agentScopeModelFingerprint(model);
        String second = service.agentScopeModelFingerprint(model);

        assertThat(first).isNotEqualTo(second);
        assertThat(first).doesNotContain("first-key", "proxy-a");
        assertThat(second).doesNotContain("second-key", "proxy-b");
    }

    private static AiProviderContext context(AiModel model, ApiConfig apiConfig) {
        return AiProviderContext.builder()
                .model(model)
                .apiConfig(apiConfig)
                .config(Map.of("temperature", 0.2))
                .platform(apiConfig.getPlatform())
                .apiKey(apiConfig.getApiKey())
                .baseUrl(apiConfig.getApiUrl())
                .modelName(model.getCode())
                .build();
    }
}
