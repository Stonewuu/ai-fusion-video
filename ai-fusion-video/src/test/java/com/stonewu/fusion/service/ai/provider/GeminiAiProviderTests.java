package com.stonewu.fusion.service.ai.provider;

import com.google.genai.types.Content;
import com.google.genai.types.FunctionResponse;
import com.google.genai.types.FunctionCallingConfig;
import com.google.genai.types.FunctionCallingConfigMode;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.Part;
import com.google.genai.types.ThinkingLevel;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import io.agentscope.core.formatter.Formatter;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolChoice;
import io.agentscope.extensions.model.gemini.GeminiChatModel;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class GeminiAiProviderTests {

    @Test
    void createAgentScopeModelUsesOfficialOptionsProxyAndToolResultOrdering() throws Exception {
        GeminiAiProvider provider = new GeminiAiProvider();
        AiProviderContext context = AiProviderContext.builder()
                .platform("gemini")
                .apiKey("test-key")
                .modelName("gemini-2.5-flash")
                .model(AiModel.builder().supportReasoning(true).build())
                .config(Map.of("temperature", 0.3))
                .apiConfig(ApiConfig.builder()
                        .proxyType("http")
                        .proxyHost("127.0.0.1")
                        .proxyPort(7890)
                        .build())
                .build();

        ChatModelBase model = provider.createAgentScopeModel(context);

        assertThat(model).isInstanceOf(GeminiChatModel.class);
        GeminiChatModel geminiModel = (GeminiChatModel) model;
        try {
            GenerateOptions options = extractDefaultOptions(geminiModel);
            assertThat(options.getThinkingBudget()).isEqualTo(-1);
            assertThat(options.getTemperature()).isEqualTo(0.3);

            OkHttpClientProbe proxyProbe = extractHttpClient(geminiModel);
            assertThat(proxyProbe.address().getHostString()).isEqualTo("127.0.0.1");
            assertThat(proxyProbe.address().getPort()).isEqualTo(7890);

            assertToolResultsFollowToolCallOrder(geminiModel);
        } finally {
            geminiModel.close();
        }
    }

    @Test
    void agentScopeFormatterDelegatesRequiredNoneAndSpecificToolChoice() {
        Formatter<Content, ?, GenerateContentConfig.Builder> formatter = GeminiAiProvider.agentScopeFormatter();

        FunctionCallingConfig required = applyToolChoice(formatter, new ToolChoice.Required());
        assertThat(required.mode().orElseThrow().knownEnum()).isEqualTo(FunctionCallingConfigMode.Known.ANY);
        assertThat(required.allowedFunctionNames()).isEmpty();

        FunctionCallingConfig none = applyToolChoice(formatter, new ToolChoice.None());
        assertThat(none.mode().orElseThrow().knownEnum()).isEqualTo(FunctionCallingConfigMode.Known.NONE);

        FunctionCallingConfig specific = applyToolChoice(formatter, new ToolChoice.Specific("get_project"));
        assertThat(specific.mode().orElseThrow().knownEnum()).isEqualTo(FunctionCallingConfigMode.Known.ANY);
        assertThat(specific.allowedFunctionNames()).contains(List.of("get_project"));
    }

    @Test
    void agentScopeFormatterMapsReasoningEffortToThinkingLevel() {
        Formatter<Content, ?, GenerateContentConfig.Builder> formatter = GeminiAiProvider.agentScopeFormatter();
        GenerateContentConfig.Builder builder = GenerateContentConfig.builder();

        formatter.applyOptions(
                builder,
                null,
                GenerateOptions.builder().reasoningEffort("minimal").build());

        var thinking = builder.build().thinkingConfig().orElseThrow();
        assertThat(thinking.includeThoughts()).contains(true);
        assertThat(thinking.thinkingLevel().orElseThrow().knownEnum())
                .isEqualTo(ThinkingLevel.Known.MINIMAL);
    }

    @Test
    void agentScopeFormatterOrdersMultipleResultsWithinOneToolMessageByBlock() {
        Formatter<Content, ?, GenerateContentConfig.Builder> formatter = GeminiAiProvider.agentScopeFormatter();
        Msg assistantToolCall = Msg.builder()
                .role(MsgRole.ASSISTANT)
                .content(
                        ToolUseBlock.builder().id("call-1").name("get_project")
                                .input(Map.of("projectId", 2)).build(),
                        ToolUseBlock.builder().id("call-2").name("get_storyboard")
                                .input(Map.of("storyboardId", 11)).build())
                .build();
        Msg combinedToolResults = Msg.builder()
                .role(MsgRole.TOOL)
                .content(
                        ToolResultBlock.text("storyboard-ok").withIdAndName("call-2", "get_storyboard"),
                        ToolResultBlock.text("extra-ok").withIdAndName("call-extra", "extra_tool"),
                        ToolResultBlock.text("project-ok").withIdAndName("call-1", "get_project"))
                .build();

        List<Content> contents = formatter.format(List.of(assistantToolCall, combinedToolResults));
        List<Part> responseParts = contents.get(1).parts().orElseThrow();
        assertThat(responseParts).hasSize(3);
        assertFunctionResponse(responseParts.get(0), "call-1", "get_project", "project-ok");
        assertFunctionResponse(responseParts.get(1), "call-2", "get_storyboard", "storyboard-ok");
        assertFunctionResponse(responseParts.get(2), "call-extra", "extra_tool", "extra-ok");
    }

    private FunctionCallingConfig applyToolChoice(
            Formatter<Content, ?, GenerateContentConfig.Builder> formatter,
            ToolChoice toolChoice) {
        GenerateContentConfig.Builder builder = GenerateContentConfig.builder();
        formatter.applyToolChoice(builder, toolChoice);
        return builder.build().toolConfig().orElseThrow().functionCallingConfig().orElseThrow();
    }

    @SuppressWarnings("unchecked")
    private void assertToolResultsFollowToolCallOrder(GeminiChatModel model) throws Exception {
        Formatter<Content, ?, ?> formatter = (Formatter<Content, ?, ?>) findField(
                GeminiChatModel.class, "formatter").get(model);
        Msg assistantToolCall = Msg.builder()
                .role(MsgRole.ASSISTANT)
                .content(
                        ToolUseBlock.builder().id("call-1").name("get_project")
                                .input(Map.of("projectId", 2)).build(),
                        ToolUseBlock.builder().id("call-2").name("get_storyboard")
                                .input(Map.of("storyboardId", 11)).build())
                .build();
        Msg secondToolResult = Msg.builder()
                .role(MsgRole.TOOL)
                .content(ToolResultBlock.text("storyboard-ok").withIdAndName("call-2", "get_storyboard"))
                .build();
        Msg firstToolResult = Msg.builder()
                .role(MsgRole.TOOL)
                .content(ToolResultBlock.text("project-ok").withIdAndName("call-1", "get_project"))
                .build();

        List<Content> contents = formatter.format(List.of(assistantToolCall, secondToolResult, firstToolResult));
        List<Part> responseParts = contents.get(1).parts().orElseThrow();
        assertFunctionResponse(responseParts.get(0), "call-1", "get_project", "project-ok");
        assertFunctionResponse(responseParts.get(1), "call-2", "get_storyboard", "storyboard-ok");
    }

    private void assertFunctionResponse(Part part, String id, String name, String output) {
        FunctionResponse response = part.functionResponse().orElseThrow();
        assertThat(response.id()).contains(id);
        assertThat(response.name()).contains(name);
        assertThat(response.response().orElseThrow().get("output")).isEqualTo(output);
    }

    private OkHttpClientProbe extractHttpClient(GeminiChatModel model) throws Exception {
        Object client = findField(GeminiChatModel.class, "client").get(model);
        Object apiClient = findField(client.getClass(), "apiClient").get(client);
        okhttp3.OkHttpClient httpClient = (okhttp3.OkHttpClient) findField(
                apiClient.getClass(), "httpClient").get(apiClient);
        Proxy proxy = httpClient.proxy();
        assertThat(proxy).isNotNull();
        return new OkHttpClientProbe((InetSocketAddress) proxy.address());
    }

    private GenerateOptions extractDefaultOptions(GeminiChatModel model) throws Exception {
        Field field = findField(GeminiChatModel.class, "defaultOptions");
        return (GenerateOptions) field.get(model);
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

    private record OkHttpClientProbe(InetSocketAddress address) {
    }
}
