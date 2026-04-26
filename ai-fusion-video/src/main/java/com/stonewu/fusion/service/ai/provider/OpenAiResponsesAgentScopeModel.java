package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.core.JsonValue;
import com.openai.core.http.StreamResponse;
import com.openai.models.Reasoning;
import com.openai.models.ReasoningEffort;
import com.openai.models.responses.EasyInputMessage;
import com.openai.models.responses.FunctionTool;
import com.openai.models.responses.ResponseCreateParams;
import com.openai.models.responses.ResponseFunctionToolCall;
import com.openai.models.responses.ResponseInputItem;
import com.openai.models.responses.ResponseOutputItem;
import com.openai.models.responses.ResponseStreamEvent;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ThinkingBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.ChatUsage;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.Model;
import io.agentscope.core.model.ToolSchema;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

@RequiredArgsConstructor
@Slf4j
public class OpenAiResponsesAgentScopeModel implements Model {

    private static final String DEFAULT_BASE_URL = "https://api.openai.com";
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final TypeReference<Map<String, Object>> MAP_TYPE_REFERENCE =
            new TypeReference<>() {};

    private final ApiConfig apiConfig;
    private final String apiKey;
    private final String baseUrl;
    private final String modelName;
    private final GenerateOptions defaultGenerateOptions;

    @Override
    public Flux<ChatResponse> stream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions runtimeOptions) {
        return Flux.create(sink -> {
            GenerateOptions effectiveOptions = mergeOptions(runtimeOptions);
            ResponseCreateParams params = buildRequestParams(messages, tools, effectiveOptions);
            Map<String, ResponseFunctionToolCall> functionCallsByItemId = new HashMap<>();
            boolean[] sawToolCall = {false};
            OpenAIClient client = buildClient();

            try (StreamResponse<ResponseStreamEvent> streamResponse = client.responses().createStreaming(params);
                 var eventStream = streamResponse.stream()) {
                var iterator = eventStream.iterator();
                while (!sink.isCancelled() && iterator.hasNext()) {
                    ResponseStreamEvent event = iterator.next();
                    rememberFunctionCall(event, functionCallsByItemId);

                    if (event.isError()) {
                        throw new RuntimeException("OpenAI Responses 流式调用失败: " + event.asError());
                    }
                    if (event.isFailed()) {
                        throw new RuntimeException("OpenAI Responses 调用失败: " + event.asFailed());
                    }
                    if (event.isIncomplete()) {
                        throw new RuntimeException("OpenAI Responses 返回未完成结果: " + event.asIncomplete());
                    }

                    event.outputTextDelta().ifPresent(textDelta -> sink.next(buildChunk(
                            List.of(TextBlock.builder().text(textDelta.delta()).build()),
                            null,
                            Map.of("responsesEvent", "output_text.delta"),
                            null)));

                    event.reasoningTextDelta().ifPresent(reasoningDelta -> sink.next(buildChunk(
                            List.of(ThinkingBlock.builder().thinking(reasoningDelta.delta()).build()),
                            null,
                            Map.of("responsesEvent", "reasoning_text.delta"),
                            null)));

                    event.reasoningSummaryTextDelta().ifPresent(reasoningDelta -> sink.next(buildChunk(
                            List.of(ThinkingBlock.builder().thinking(reasoningDelta.delta()).build()),
                            null,
                            Map.of("responsesEvent", "reasoning_summary_text.delta"),
                            null)));

                    event.functionCallArgumentsDone().ifPresent(toolCallDone -> {
                        ResponseFunctionToolCall functionCall = functionCallsByItemId.get(toolCallDone.itemId());
                        String toolCallId = functionCall != null ? functionCall.callId() : toolCallDone.itemId();
                        String toolName = functionCall != null ? functionCall.name() : toolCallDone.name();
                        String arguments = functionCall != null ? functionCall.arguments() : toolCallDone.arguments();
                        sawToolCall[0] = true;
                        sink.next(buildChunk(
                                List.of(ToolUseBlock.builder()
                                        .id(toolCallId)
                                        .name(toolName)
                                        .input(parseArguments(arguments))
                                        .content(arguments)
                                        .build()),
                                null,
                                Map.of("responsesEvent", "function_call_arguments.done"),
                                null));
                    });

                    event.completed().ifPresent(completed -> sink.next(buildChunk(
                            List.of(),
                            buildUsage(completed.response().usage().orElse(null)),
                            Map.of("responsesEvent", "completed"),
                            sawToolCall[0] ? "tool_calls" : "stop")));
                }
                sink.complete();
            } catch (Exception e) {
                log.warn("OpenAI Responses 调用失败: model={}, message={}", getModelName(), e.getMessage());
                sink.error(e);
            } finally {
                client.close();
            }
        });
    }

    @Override
    public String getModelName() {
        return modelName;
    }

    ResponseCreateParams buildRequestParams(List<Msg> messages, List<ToolSchema> tools, GenerateOptions runtimeOptions) {
        GenerateOptions effectiveOptions = mergeOptions(runtimeOptions);
        ResponseCreateParams.Builder builder = ResponseCreateParams.builder()
                .model(resolveModelName(effectiveOptions));

        List<ResponseInputItem> inputItems = mapMessages(messages);
        if (inputItems.isEmpty()) {
            builder.input("");
        } else {
            builder.inputOfResponse(inputItems);
        }

        for (FunctionTool tool : mapTools(tools)) {
            builder.addTool(tool);
        }

        applyGenerateOptions(builder, effectiveOptions);
        return builder.build();
    }

    List<ResponseInputItem> mapMessages(List<Msg> messages) {
        List<ResponseInputItem> inputItems = new ArrayList<>();
        if (messages == null) {
            return inputItems;
        }

        for (Msg message : messages) {
            if (message == null || message.getRole() == null) {
                continue;
            }

            if (message.getRole() == MsgRole.TOOL) {
                for (ToolResultBlock block : message.getContentBlocks(ToolResultBlock.class)) {
                    if (StrUtil.isBlank(block.getId())) {
                        continue;
                    }
                    inputItems.add(ResponseInputItem.ofFunctionCallOutput(
                            ResponseInputItem.FunctionCallOutput.builder()
                                    .callId(block.getId())
                                    .output(extractToolOutput(block))
                                    .build()));
                }
                continue;
            }

            if (message.getRole() == MsgRole.ASSISTANT) {
                appendAssistantMessage(message, inputItems);
                continue;
            }

            String textContent = message.getTextContent();
            if (StrUtil.isNotBlank(textContent)) {
                inputItems.add(ResponseInputItem.ofEasyInputMessage(EasyInputMessage.builder()
                        .role(mapRole(message.getRole()))
                        .content(textContent)
                        .build()));
            }
        }
        return inputItems;
    }

    List<FunctionTool> mapTools(List<ToolSchema> tools) {
        List<FunctionTool> responseTools = new ArrayList<>();
        if (tools == null) {
            return responseTools;
        }

        for (ToolSchema tool : tools) {
            if (tool == null || StrUtil.isBlank(tool.getName())) {
                continue;
            }

            FunctionTool.Builder builder = FunctionTool.builder().name(tool.getName());
            if (StrUtil.isNotBlank(tool.getDescription())) {
                builder.description(tool.getDescription());
            }
            if (tool.getStrict() != null) {
                builder.strict(tool.getStrict());
            }

            FunctionTool.Parameters.Builder parametersBuilder = FunctionTool.Parameters.builder();
            Map<String, Object> parameters = tool.getParameters();
            if (parameters != null) {
                parameters.forEach((key, value) -> parametersBuilder.putAdditionalProperty(key, JsonValue.from(value)));
            }
            builder.parameters(parametersBuilder.build());
            responseTools.add(builder.build());
        }
        return responseTools;
    }

    Reasoning buildReasoning(GenerateOptions options) {
        if (options == null) {
            return null;
        }

        Reasoning.Builder builder = Reasoning.builder();
        boolean hasReasoning = false;

        if (StrUtil.isNotBlank(options.getReasoningEffort())) {
            builder.effort(ReasoningEffort.of(options.getReasoningEffort().toLowerCase(Locale.ROOT)));
            hasReasoning = true;
        }

        String summaryMode = getBodyParamString(options, "reasoningSummary", "reasoning_summary");
        Boolean includeReasoning = getBodyParamBoolean(options, "includeReasoning", "include_reasoning");
        if (StrUtil.isBlank(summaryMode) && Boolean.TRUE.equals(includeReasoning)) {
            summaryMode = "auto";
        }
        if (StrUtil.isNotBlank(summaryMode)) {
            builder.summary(Reasoning.Summary.of(summaryMode.toLowerCase(Locale.ROOT)));
            hasReasoning = true;
        }

        return hasReasoning ? builder.build() : null;
    }

    private OpenAIClient buildClient() {
        OpenAIOkHttpClient.Builder builder = OpenAIOkHttpClient.builder()
                .apiKey(apiKey)
                .baseUrl(resolveOpenAiJavaBaseUrl())
                .timeout(Duration.ofMinutes(3))
                .maxRetries(2)
                .responseValidation(false);

        if (apiConfig != null) {
            java.net.Proxy proxy = AiProxySupport.javaProxyOrNull(apiConfig);
            if (proxy != null) {
                builder.proxy(proxy);
            }
        }

        return builder.build();
    }

    private void applyGenerateOptions(ResponseCreateParams.Builder builder, GenerateOptions options) {
        if (options == null) {
            return;
        }

        if (options.getTemperature() != null) {
            builder.temperature(options.getTemperature());
        }
        if (options.getTopP() != null) {
            builder.topP(options.getTopP());
        }

        Integer maxOutputTokens = options.getMaxCompletionTokens() != null
                ? options.getMaxCompletionTokens() : options.getMaxTokens();
        if (maxOutputTokens != null) {
            builder.maxOutputTokens(Long.valueOf(maxOutputTokens));
        }

        if (options.getAdditionalHeaders() != null) {
            options.getAdditionalHeaders().forEach(builder::putAdditionalHeader);
        }
        if (options.getAdditionalQueryParams() != null) {
            options.getAdditionalQueryParams().forEach(builder::putAdditionalQueryParam);
        }

        Reasoning reasoning = buildReasoning(options);
        if (reasoning != null) {
            builder.reasoning(reasoning);
        }

        if (options.getAdditionalBodyParams() != null) {
            options.getAdditionalBodyParams().forEach((key, value) -> {
                if (value == null || isHandledReasoningKey(key)) {
                    return;
                }
                builder.putAdditionalBodyProperty(key, JsonValue.from(value));
            });
        }
    }

    private void appendAssistantMessage(Msg message, List<ResponseInputItem> inputItems) {
        StringBuilder textBuffer = new StringBuilder();
        for (ContentBlock block : message.getContent()) {
            if (block instanceof TextBlock textBlock) {
                textBuffer.append(textBlock.getText());
                continue;
            }
            if (block instanceof ThinkingBlock thinkingBlock) {
                textBuffer.append(thinkingBlock.getThinking());
                continue;
            }
            if (block instanceof ToolUseBlock toolUseBlock) {
                flushAssistantText(textBuffer, inputItems);
                inputItems.add(ResponseInputItem.ofFunctionCall(ResponseFunctionToolCall.builder()
                        .callId(StrUtil.blankToDefault(toolUseBlock.getId(), toolUseBlock.getName()))
                        .name(toolUseBlock.getName())
                        .arguments(toJson(toolUseBlock.getInput()))
                        .build()));
            }
        }
        flushAssistantText(textBuffer, inputItems);
    }

    private void flushAssistantText(StringBuilder textBuffer, List<ResponseInputItem> inputItems) {
        if (textBuffer.isEmpty()) {
            return;
        }
        inputItems.add(ResponseInputItem.ofEasyInputMessage(EasyInputMessage.builder()
                .role(EasyInputMessage.Role.ASSISTANT)
                .content(textBuffer.toString())
                .build()));
        textBuffer.setLength(0);
    }

    private void rememberFunctionCall(ResponseStreamEvent event, Map<String, ResponseFunctionToolCall> functionCallsByItemId) {
        event.outputItemAdded().ifPresent(outputItem -> rememberFunctionCall(outputItem.item(), functionCallsByItemId));
        event.outputItemDone().ifPresent(outputItem -> rememberFunctionCall(outputItem.item(), functionCallsByItemId));
    }

    private void rememberFunctionCall(ResponseOutputItem outputItem,
                                      Map<String, ResponseFunctionToolCall> functionCallsByItemId) {
        if (outputItem == null || !outputItem.isFunctionCall()) {
            return;
        }
        ResponseFunctionToolCall functionCall = outputItem.asFunctionCall();
        String itemId = functionCall.id().orElse(null);
        if (StrUtil.isNotBlank(itemId)) {
            functionCallsByItemId.put(itemId, functionCall);
        }
    }

    private ChatResponse buildChunk(List<? extends ContentBlock> content,
                                    ChatUsage usage,
                                    Map<String, Object> metadata,
                                    String finishReason) {
        ChatResponse.Builder builder = ChatResponse.builder()
                .content(content == null ? List.of() : List.copyOf(content));
        if (usage != null) {
            builder.usage(usage);
        }
        if (metadata != null && !metadata.isEmpty()) {
            builder.metadata(metadata);
        }
        if (StrUtil.isNotBlank(finishReason)) {
            builder.finishReason(finishReason);
        }
        return builder.build();
    }

    private ChatUsage buildUsage(com.openai.models.responses.ResponseUsage usage) {
        if (usage == null) {
            return null;
        }
        return new ChatUsage((int) usage.inputTokens(), (int) usage.outputTokens(), 0d);
    }

    private String resolveOpenAiJavaBaseUrl() {
        String normalized = normalizeBaseUrl(StrUtil.blankToDefault(baseUrl, DEFAULT_BASE_URL));
        if (endsWithIgnoreCase(normalized, "/v1")) {
            return normalized;
        }
        if (shouldAutoAppendV1Path()) {
            return normalized + "/v1";
        }
        return normalized;
    }

    private boolean shouldAutoAppendV1Path() {
        if (apiConfig == null) {
            return true;
        }
        if (!"openai_compatible".equalsIgnoreCase(apiConfig.getPlatform())) {
            return true;
        }
        return !Boolean.FALSE.equals(apiConfig.getAutoAppendV1Path());
    }

    private String resolveModelName(GenerateOptions options) {
        return options != null && StrUtil.isNotBlank(options.getModelName())
                ? options.getModelName() : modelName;
    }

    private GenerateOptions mergeOptions(GenerateOptions runtimeOptions) {
        if (defaultGenerateOptions == null) {
            return runtimeOptions;
        }
        if (runtimeOptions == null) {
            return defaultGenerateOptions;
        }
        return GenerateOptions.mergeOptions(defaultGenerateOptions, runtimeOptions);
    }

    private EasyInputMessage.Role mapRole(MsgRole role) {
        if (role == MsgRole.SYSTEM) {
            return EasyInputMessage.Role.SYSTEM;
        }
        if (role == MsgRole.ASSISTANT) {
            return EasyInputMessage.Role.ASSISTANT;
        }
        return EasyInputMessage.Role.USER;
    }

    private Map<String, Object> parseArguments(String arguments) {
        if (StrUtil.isBlank(arguments)) {
            return Map.of();
        }
        try {
            return OBJECT_MAPPER.readValue(arguments, MAP_TYPE_REFERENCE);
        } catch (Exception ignored) {
            return Map.of("raw", arguments);
        }
    }

    private String extractToolOutput(ToolResultBlock block) {
        StringBuilder builder = new StringBuilder();
        for (ContentBlock output : block.getOutput()) {
            if (output instanceof TextBlock textBlock) {
                builder.append(textBlock.getText());
            } else if (output instanceof ThinkingBlock thinkingBlock) {
                builder.append(thinkingBlock.getThinking());
            }
        }
        return builder.toString();
    }

    private String toJson(Object value) {
        try {
            return OBJECT_MAPPER.writeValueAsString(value == null ? Map.of() : value);
        } catch (Exception e) {
            return "{}";
        }
    }

    private String getBodyParamString(GenerateOptions options, String... keys) {
        if (options == null || options.getAdditionalBodyParams() == null) {
            return null;
        }
        for (String key : keys) {
            Object value = options.getAdditionalBodyParams().get(key);
            if (value instanceof String stringValue && StrUtil.isNotBlank(stringValue)) {
                return stringValue;
            }
        }
        return null;
    }

    private Boolean getBodyParamBoolean(GenerateOptions options, String... keys) {
        if (options == null || options.getAdditionalBodyParams() == null) {
            return null;
        }
        for (String key : keys) {
            Object value = options.getAdditionalBodyParams().get(key);
            if (value instanceof Boolean booleanValue) {
                return booleanValue;
            }
            if (value instanceof String stringValue && StrUtil.isNotBlank(stringValue)) {
                return Boolean.parseBoolean(stringValue);
            }
        }
        return null;
    }

    private boolean isHandledReasoningKey(String key) {
        if (key == null) {
            return false;
        }
        return switch (key) {
            case "includeReasoning", "include_reasoning", "reasoningSummary", "reasoning_summary" -> true;
            default -> false;
        };
    }

    private String normalizeBaseUrl(String rawBaseUrl) {
        return StrUtil.blankToDefault(rawBaseUrl, DEFAULT_BASE_URL).trim().replaceAll("/+$", "");
    }

    private boolean endsWithIgnoreCase(String text, String suffix) {
        return text != null && suffix != null && text.toLowerCase(Locale.ROOT)
                .endsWith(suffix.toLowerCase(Locale.ROOT));
    }
}