package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.transport.OkHttpTransport;
import io.agentscope.extensions.model.dashscope.DashScopeChatModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class DashScopeAiProviderTests {

    @Test
    void createAgentScopeModelUsesOfficialOptionsAndProxy() throws Exception {
        DashScopeAiProvider provider = new DashScopeAiProvider();
        AiProviderContext context = AiProviderContext.builder()
                .platform("dashscope")
                .apiKey("test-key")
                .baseUrl("https://dashscope.aliyuncs.com")
                .modelName("qwen-plus")
                .model(AiModel.builder().supportReasoning(true).build())
                .config(Map.of("temperature", 0.4, "topP", 0.7))
                .apiConfig(ApiConfig.builder()
                        .proxyType("http")
                        .proxyHost("127.0.0.1")
                        .proxyPort(7890)
                        .build())
                .build();

        ChatModelBase model = provider.createAgentScopeModel(context);

        assertThat(model).isInstanceOf(DashScopeChatModel.class);
        assertThat(model.getModelName()).isEqualTo("qwen-plus");
        GenerateOptions options = (GenerateOptions) readField(model, "defaultOptions");
        assertThat(options.getTemperature()).isEqualTo(0.4);
        assertThat(options.getTopP()).isEqualTo(0.7);
        assertThat(readField(model, "enableThinking")).isEqualTo(true);

        Object httpClient = readField(model, "httpClient");
        OkHttpTransport transport = (OkHttpTransport) readField(httpClient, "transport");
        assertThat(transport.getConfig().getProxyConfig().getHost()).isEqualTo("127.0.0.1");
        assertThat(transport.getConfig().getProxyConfig().getPort()).isEqualTo(7890);
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
