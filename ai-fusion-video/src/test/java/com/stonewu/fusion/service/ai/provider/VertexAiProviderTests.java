package com.stonewu.fusion.service.ai.provider;

import com.google.auth.oauth2.GoogleCredentials;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.extensions.model.gemini.GeminiChatModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class VertexAiProviderTests {

    @Test
    void createAgentScopeModelUsesOfficialVertexCredentialsOptionsAndProxy() throws Exception {
        String authorizedUserCredentials = """
                {"type":"authorized_user","client_id":"client-id","client_secret":"client-secret","refresh_token":"refresh-token"}
                """;
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("vertex_ai")
                .appId("demo-project")
                .appSecret(authorizedUserCredentials)
                .proxyType("http")
                .proxyHost("127.0.0.1")
                .proxyPort(7890)
                .build();
        AiProviderContext context = AiProviderContext.builder()
                .platform("vertex_ai")
                .modelName("gemini-2.5-flash")
                .model(AiModel.builder().supportReasoning(true).build())
                .config(Map.of("projectId", "demo-project", "location", "global", "temperature", 0.25))
                .apiConfig(apiConfig)
                .build();

        ChatModelBase model = new VertexAiProvider().createAgentScopeModel(context);

        assertThat(model).isInstanceOf(GeminiChatModel.class);
        GeminiChatModel geminiModel = (GeminiChatModel) model;
        try {
            assertThat(readField(geminiModel, "project")).isEqualTo("demo-project");
            assertThat(readField(geminiModel, "location")).isEqualTo("global");
            assertThat(readField(geminiModel, "vertexAI")).isEqualTo(true);
            assertThat(readField(geminiModel, "streamEnabled")).isEqualTo(true);
            assertThat(readField(geminiModel, "credentials")).isInstanceOf(GoogleCredentials.class);

            GenerateOptions options = (GenerateOptions) readField(geminiModel, "defaultOptions");
            assertThat(options.getThinkingBudget()).isEqualTo(-1);
            assertThat(options.getTemperature()).isEqualTo(0.25);

            Object client = readField(geminiModel, "client");
            Object apiClient = readField(client, "apiClient");
            okhttp3.OkHttpClient httpClient = (okhttp3.OkHttpClient) readField(apiClient, "httpClient");
            Proxy proxy = httpClient.proxy();
            assertThat(proxy).isNotNull();
            InetSocketAddress address = (InetSocketAddress) proxy.address();
            assertThat(address.getHostString()).isEqualTo("127.0.0.1");
            assertThat(address.getPort()).isEqualTo(7890);
        } finally {
            geminiModel.close();
        }
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
