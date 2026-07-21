package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.transport.ProxyConfig;
import io.agentscope.extensions.model.anthropic.AnthropicChatModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.net.InetSocketAddress;
import java.net.Proxy;
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

        ChatModelBase model = provider.createAgentScopeModel(context);

        assertThat(model).isInstanceOf(AnthropicChatModel.class);
        assertThat(model.getModelName()).isEqualTo("claude-sonnet-4-5-20250929");
    }

    @Test
    void createAgentScopeModelUsesOfficialOptionsAndProxyAddressWhilePreservingCredentialConfig() throws Exception {
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
                        .proxyUsername("proxy-user")
                        .proxyPassword("proxy-pass")
                        .build())
                .build();

        ChatModelBase model = provider.createAgentScopeModel(context);

        assertThat(model).isInstanceOf(AnthropicChatModel.class);
        assertThat(model.getModelName()).isEqualTo("claude-sonnet-4-5-20250929");
        GenerateOptions options = (GenerateOptions) readField(model, "defaultOptions");
        assertThat(options.getThinkingBudget()).isEqualTo(2048);

        Object client = readField(model, "client");
        Object clientOptions = readField(client, "clientOptions");
        Object httpClient = clientOptions.getClass().getMethod("httpClient").invoke(clientOptions);
        while (!"com.anthropic.client.okhttp.OkHttpClient".equals(httpClient.getClass().getName())) {
            httpClient = readField(httpClient, "httpClient");
        }
        okhttp3.OkHttpClient okHttpClient = (okhttp3.OkHttpClient) readField(httpClient, "okHttpClient");
        Proxy proxy = okHttpClient.proxy();
        assertThat(proxy).isNotNull();
        InetSocketAddress address = (InetSocketAddress) proxy.address();
        assertThat(address.getHostString()).isEqualTo("127.0.0.1");
        assertThat(address.getPort()).isEqualTo(7890);

        // AgentScope GA currently exposes only the proxy address on the Anthropic SDK client.
        // These assertions cover credential preservation at our config boundary, not end-to-end auth.
        ProxyConfig proxyConfig = AiProxySupport.agentScopeProxyConfig(context.getApiConfig());
        assertThat(proxyConfig.hasAuthentication()).isTrue();
        assertThat(proxyConfig.getUsername()).isEqualTo("proxy-user");
        assertThat(proxyConfig.getPassword()).isEqualTo("proxy-pass");
    }

    private Object readField(Object target, String fieldName) throws ReflectiveOperationException {
        Field field = findField(target.getClass(), fieldName);
        return field.get(target);
    }

    private Field findField(Class<?> type, String fieldName) throws NoSuchFieldException {
        Class<?> current = type;
        while (current != null) {
            try {
                Field field = current.getDeclaredField(fieldName);
                field.setAccessible(true);
                return field;
            } catch (NoSuchFieldException ignored) {
                current = current.getSuperclass();
            }
        }
        throw new NoSuchFieldException(fieldName);
    }
}
