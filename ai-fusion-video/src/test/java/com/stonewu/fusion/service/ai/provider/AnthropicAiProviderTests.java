package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.ApiConfig;
import io.agentscope.core.model.AnthropicChatModel;
import io.agentscope.core.model.Model;
import io.agentscope.core.model.transport.ProxyConfig;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class AnthropicAiProviderTests {

    @Test
    void createAgentScopeModelKeepsOfficialModelWithoutProxy() {
        AnthropicAiProvider provider = new AnthropicAiProvider();

        AiProviderContext context = AiProviderContext.builder()
                .platform("anthropic")
                .apiKey("test-key")
                .baseUrl("https://api.anthropic.com")
                .modelName("claude-sonnet-4-5-20250929")
                .apiConfig(ApiConfig.builder()
                        .platform("anthropic")
                        .apiUrl("https://api.anthropic.com")
                        .build())
                .build();

        Model model = provider.createAgentScopeModel(context);

        assertThat(model).isInstanceOf(AnthropicChatModel.class);
        assertThat(model.getModelName()).isEqualTo("claude-sonnet-4-5-20250929");
    }

    @Test
    void createAgentScopeModelSupportsProxyEnabledAnthropicModels() {
        AnthropicAiProvider provider = new AnthropicAiProvider();

        AiProviderContext context = AiProviderContext.builder()
                .platform("anthropic")
                .apiKey("test-key")
                .baseUrl("https://api.anthropic.com")
                .modelName("claude-sonnet-4-5-20250929")
                .config(Map.of("thinkingBudget", 2048))
                .apiConfig(ApiConfig.builder()
                        .platform("anthropic")
                        .apiUrl("https://api.anthropic.com")
                        .proxyType("http")
                        .proxyHost("127.0.0.1")
                        .proxyPort(7890)
                        .build())
                .build();

        Model model = provider.createAgentScopeModel(context);
        boolean officialProxySupported = Arrays.stream(AnthropicChatModel.Builder.class.getMethods())
                .anyMatch(method -> method.getName().equals("proxy")
                        && method.getParameterCount() == 1
                        && method.getParameterTypes()[0] == ProxyConfig.class);

        assertThat(model).isInstanceOf(officialProxySupported
                ? AnthropicChatModel.class
                : ProxyAwareAnthropicChatModel.class);
        assertThat(model.getModelName()).isEqualTo("claude-sonnet-4-5-20250929");
    }
}