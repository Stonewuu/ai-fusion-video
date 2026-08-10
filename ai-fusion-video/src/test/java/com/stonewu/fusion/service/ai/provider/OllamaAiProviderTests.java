package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.ApiConfig;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.transport.OkHttpTransport;
import io.agentscope.extensions.model.ollama.OllamaChatModel;
import io.agentscope.extensions.model.ollama.options.OllamaOptions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class OllamaAiProviderTests {

    @Test
    void createAgentScopeModelUsesOfficialOptionsAndProxy() throws Exception {
        OllamaAiProvider provider = new OllamaAiProvider();
        AiProviderContext context = AiProviderContext.builder()
                .platform("ollama")
                .baseUrl("http://localhost:11434")
                .modelName("qwen3:8b")
                .config(Map.of("temperature", 0.5, "topP", 0.6))
                .apiConfig(ApiConfig.builder()
                        .proxyType("socks5")
                        .proxyHost("127.0.0.1")
                        .proxyPort(1080)
                        .build())
                .build();

        ChatModelBase model = provider.createAgentScopeModel(context);

        assertThat(model).isInstanceOf(OllamaChatModel.class);
        assertThat(model.getModelName()).isEqualTo("qwen3:8b");
        OllamaOptions options = (OllamaOptions) readField(model, "defaultOptions");
        assertThat(options.getTemperature()).isEqualTo(0.5);
        assertThat(options.getTopP()).isEqualTo(0.6);

        Object httpClient = readField(model, "httpClient");
        OkHttpTransport transport = (OkHttpTransport) readField(httpClient, "transport");
        assertThat(transport.getConfig().getProxyConfig().getHost()).isEqualTo("127.0.0.1");
        assertThat(transport.getConfig().getProxyConfig().getPort()).isEqualTo(1080);
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
