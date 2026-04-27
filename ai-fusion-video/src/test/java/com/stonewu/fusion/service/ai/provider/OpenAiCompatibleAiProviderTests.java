package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.ApiConfig;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.Model;
import org.junit.jupiter.api.Test;

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

        Model model = provider.createAgentScopeModel(context);

        assertThat(model).isInstanceOf(OpenAiResponsesAgentScopeModel.class);
        assertThat(model.getModelName()).isEqualTo("gpt-5");
    }

    @Test
    void createAgentScopeModelKeepsChatCompletionsByDefault() {
        OpenAiCompatibleAiProvider provider = new OpenAiCompatibleAiProvider();

        AiProviderContext context = AiProviderContext.builder()
                                .platform("openai_compatible")
                .apiKey("test-key")
                .baseUrl("https://api.openai.com")
                .modelName("gpt-4o-mini")
                                .apiConfig(ApiConfig.builder().platform("openai_compatible").apiUrl("https://api.openai.com").build())
                .build();

        Model model = provider.createAgentScopeModel(context);

        assertThat(model).isNotInstanceOf(OpenAiResponsesAgentScopeModel.class);
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
}