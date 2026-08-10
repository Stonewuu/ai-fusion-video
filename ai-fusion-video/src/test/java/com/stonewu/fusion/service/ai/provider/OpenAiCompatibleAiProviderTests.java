package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.ApiConfig;
import io.agentscope.core.message.Base64Source;
import io.agentscope.core.message.DataBlock;
import io.agentscope.core.message.ImageBlock;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.transport.OkHttpTransport;
import io.agentscope.extensions.model.openai.OpenAIChatModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class OpenAiCompatibleAiProviderTests {

    @Test
    void createAgentScopeModelUsesResponsesModelWhenEnabled() {
        OpenAiCompatibleAiProvider provider = new OpenAiCompatibleAiProvider();

        AiProviderContext context = AiProviderContext.builder()
                                .platform("openai_compatible")
                .apiKey("test-key")
                .baseUrl("https://api.openai.com")
                .modelName("gpt-5")
                .config(Map.of("apiMode", "responses", "reasoningEffort", "medium"))
                                .apiConfig(ApiConfig.builder().platform("openai_compatible").apiUrl("https://api.openai.com").build())
                .build();

        ChatModelBase model = provider.createAgentScopeModel(context);

        assertThat(model).isInstanceOf(OpenAiResponsesAgentScopeModel.class);
        assertThat(model.getModelName()).isEqualTo("gpt-5");
    }

    @Test
    void createAgentScopeModelBuildsOfficialChatCompletionsWithOptionsAndProxy() throws Exception {
        OpenAiCompatibleAiProvider provider = new OpenAiCompatibleAiProvider();

        AiProviderContext context = AiProviderContext.builder()
                                .platform("openai_compatible")
                .apiKey("test-key")
                .baseUrl("https://api.openai.com")
                .modelName("gpt-4o-mini")
                .config(Map.of("temperature", 0.2, "topP", 0.8))
                .apiConfig(ApiConfig.builder()
                        .platform("openai_compatible")
                        .apiUrl("https://api.openai.com")
                        .proxyType("http")
                        .proxyHost("127.0.0.1")
                        .proxyPort(7890)
                        .build())
                .build();

        ChatModelBase model = provider.createAgentScopeModel(context);

        assertThat(model).isInstanceOf(OpenAIChatModel.class);
        assertThat(model.getModelName()).isEqualTo("gpt-4o-mini");
        GenerateOptions options = (GenerateOptions) readField(model, "configuredOptions");
        assertThat(options.getTemperature()).isEqualTo(0.2);
        assertThat(options.getTopP()).isEqualTo(0.8);

        Object client = readField(model, "client");
        OkHttpTransport transport = (OkHttpTransport) client.getClass().getMethod("getTransport").invoke(client);
        assertThat(transport.getConfig().getProxyConfig().getHost()).isEqualTo("127.0.0.1");
        assertThat(transport.getConfig().getProxyConfig().getPort()).isEqualTo(7890);
    }

    @Test
    void responsesModelMapsMessagesToolsAndReasoningOptions() {
        OpenAiResponsesAgentScopeModel model = new OpenAiResponsesAgentScopeModel(
                                ApiConfig.builder().platform("openai_compatible").apiUrl("https://api.openai.com").build(),
                "test-key",
                "https://api.openai.com",
                "gpt-5",
                GenerateOptions.builder()
                        .reasoningEffort("medium")
                        .additionalBodyParam("include_reasoning", true)
                        .build());

        var systemMessage = io.agentscope.core.message.Msg.builder()
                .role(io.agentscope.core.message.MsgRole.SYSTEM)
                .content(java.util.List.of(io.agentscope.core.message.TextBlock.builder().text("You are helpful.").build()))
                .build();
        var userMessage = io.agentscope.core.message.Msg.builder()
                .role(io.agentscope.core.message.MsgRole.USER)
                .content(java.util.List.of(io.agentscope.core.message.TextBlock.builder().text("Tell me a joke.").build()))
                .build();
        var assistantMessage = io.agentscope.core.message.Msg.builder()
                .role(io.agentscope.core.message.MsgRole.ASSISTANT)
                .content(java.util.List.of(
                        io.agentscope.core.message.TextBlock.builder().text("Calling tool").build(),
                        io.agentscope.core.message.ToolUseBlock.builder()
                                .id("call_1")
                                .name("get_weather")
                                .input(Map.of("city", "Shanghai"))
                                .content("{\"city\":\"Shanghai\"}")
                                .build()))
                .build();
        var toolMessage = io.agentscope.core.message.Msg.builder()
                .role(io.agentscope.core.message.MsgRole.TOOL)
                .content(java.util.List.of(io.agentscope.core.message.ToolResultBlock.of(
                        "call_1",
                        "get_weather",
                        io.agentscope.core.message.TextBlock.builder().text("Sunny").build())))
                .build();

        var toolSchema = io.agentscope.core.model.ToolSchema.builder()
                .name("get_weather")
                .description("Get current weather")
                .parameters(Map.of("type", "object", "properties", Map.of("city", Map.of("type", "string"))))
                .strict(true)
                .build();

        var request = model.buildRequestParams(
                java.util.List.of(systemMessage, userMessage, assistantMessage, toolMessage),
                java.util.List.of(toolSchema),
                null);

        assertThat(request.model()).isPresent();
        assertThat(request.model().get().asString()).isEqualTo("gpt-5");
        assertThat(model.mapMessages(java.util.List.of(systemMessage, userMessage, assistantMessage, toolMessage))).hasSize(5);
        assertThat(model.mapTools(java.util.List.of(toolSchema))).hasSize(1);
        assertThat(model.buildReasoning(GenerateOptions.builder()
                .reasoningEffort("medium")
                .additionalBodyParam("include_reasoning", true)
                .build())).isNotNull();
    }

    @Test
    void responsesModelPreservesTypedImageBlocks() {
        OpenAiResponsesAgentScopeModel model = new OpenAiResponsesAgentScopeModel(
                ApiConfig.builder().platform("openai_compatible").apiUrl("https://api.openai.com").build(),
                "test-key",
                "https://api.openai.com",
                "gpt-5.6-sol",
                null);
        UserMessage userMessage = new UserMessage(
                TextBlock.builder().text("Describe this image").build(),
                ImageBlock.builder()
                        .source(Base64Source.builder()
                                .mediaType("image/png")
                                .data("aGVsbG8=")
                                .build())
                        .build());

        var mapped = model.mapMessages(List.of(userMessage));

        assertThat(mapped).singleElement().satisfies(item -> {
            assertThat(item.isMessage()).isTrue();
            assertThat(item.asMessage().content()).hasSize(2);
            assertThat(item.asMessage().content().get(0).isInputText()).isTrue();
            assertThat(item.asMessage().content().get(1).isInputImage()).isTrue();
            assertThat(item.asMessage().content().get(1).asInputImage().imageUrl())
                    .contains("data:image/png;base64,aGVsbG8=");
        });
    }

    @Test
    void responsesModelPreservesFileDataBlocks() {
        OpenAiResponsesAgentScopeModel model = new OpenAiResponsesAgentScopeModel(
                ApiConfig.builder().platform("openai_compatible").apiUrl("https://api.openai.com").build(),
                "test-key",
                "https://api.openai.com",
                "gpt-5.6-sol",
                null);
        UserMessage userMessage = new UserMessage(
                TextBlock.builder().text("Read this file").build(),
                DataBlock.builder()
                        .id("file-1")
                        .name("sample.pdf")
                        .source(Base64Source.builder()
                                .mediaType("application/pdf")
                                .data("aGVsbG8=")
                                .build())
                        .build());

        var mapped = model.mapMessages(List.of(userMessage));

        assertThat(mapped).singleElement().satisfies(item -> {
            assertThat(item.isMessage()).isTrue();
            assertThat(item.asMessage().content()).hasSize(2);
            assertThat(item.asMessage().content().get(1).isInputFile()).isTrue();
        });
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
