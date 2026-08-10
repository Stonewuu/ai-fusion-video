package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.google.genai.Client;
import com.google.genai.types.Content;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;
import com.google.genai.types.ThinkingConfig;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.formatter.Formatter;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolChoice;
import io.agentscope.core.model.transport.ProxyConfig;
import io.agentscope.extensions.model.gemini.GeminiChatModel;
import io.agentscope.extensions.model.gemini.formatter.GeminiChatFormatter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.google.genai.GoogleGenAiChatModel;
import org.springframework.ai.google.genai.GoogleGenAiChatOptions;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Gemini Developer API 提供商。
 */
@Component
@Slf4j
public class GeminiAiProvider extends AbstractAiProvider {

    @Override
    public boolean supports(String platform) {
        if (platform == null) {
            return false;
        }
        return "gemini".equals(platform.trim().toLowerCase(Locale.ROOT));
    }

    @Override
    public ChatModel createChatModel(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "Gemini");

        GoogleGenAiChatOptions.Builder optionsBuilder = GoogleGenAiChatOptions.builder()
                .model(context.getModelName());
        applyDouble(context.getConfig(), "temperature", optionsBuilder::temperature);

        Client genAiClient = Client.builder()
                .apiKey(context.getApiKey())
                .build();

        return GoogleGenAiChatModel.builder()
                .genAiClient(genAiClient)
                .defaultOptions(optionsBuilder.build())
                .build();
    }

    @Override
    public ChatModelBase createAgentScopeModel(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "Gemini");
        GenerateOptions defaultOptions = buildGeminiGenerateOptions(context);

        GeminiChatModel.Builder builder = GeminiChatModel.builder()
                .apiKey(context.getApiKey())
                .modelName(context.getModelName())
                .formatter(agentScopeFormatter())
                .streamEnabled(true)
                .vertexAI(false);

        if (defaultOptions != null) {
            builder.defaultOptions(defaultOptions);
        }
        ProxyConfig proxy = AiProxySupport.agentScopeProxyConfig(context.getApiConfig());
        if (proxy != null) {
            builder.proxy(proxy);
        }

        return builder.build();
    }

    static Formatter<Content, GenerateContentResponse, GenerateContentConfig.Builder> agentScopeFormatter() {
        return new ToolCallOrderedGeminiFormatter();
    }

    private static final class ToolCallOrderedGeminiFormatter
            implements Formatter<Content, GenerateContentResponse, GenerateContentConfig.Builder> {

        private final GeminiChatFormatter delegate = new GeminiChatFormatter();

        @Override
        public List<Content> format(List<Msg> messages) {
            return mergeToolResponseTurns(delegate.format(orderParallelToolResults(messages)));
        }

        @Override
        public io.agentscope.core.model.ChatResponse parseResponse(GenerateContentResponse response, Instant startTime) {
            return delegate.parseResponse(response, startTime);
        }

        @Override
        public void applyOptions(GenerateContentConfig.Builder builder, GenerateOptions options,
                                 GenerateOptions defaultOptions) {
            delegate.applyOptions(builder, options, defaultOptions);
            GenerateOptions merged = GenerateOptions.mergeOptions(options, defaultOptions);
            if (merged != null && StrUtil.isNotBlank(merged.getReasoningEffort())) {
                builder.thinkingConfig(ThinkingConfig.builder()
                        .includeThoughts(true)
                        .thinkingLevel(merged.getReasoningEffort().trim())
                        .build());
            }
        }

        @Override
        public void applyTools(GenerateContentConfig.Builder builder,
                               List<io.agentscope.core.model.ToolSchema> tools) {
            delegate.applyTools(builder, tools);
        }

        @Override
        public void applyToolChoice(GenerateContentConfig.Builder builder, ToolChoice toolChoice) {
            delegate.applyToolChoice(builder, toolChoice);
        }

        private List<Msg> orderParallelToolResults(List<Msg> messages) {
            if (messages == null || messages.size() < 2) {
                return messages;
            }
            List<Msg> ordered = new ArrayList<>(messages.size());
            for (int index = 0; index < messages.size(); index++) {
                Msg message = messages.get(index);
                ordered.add(message);
                List<ToolUseBlock> toolUses = message.getRole() == MsgRole.ASSISTANT
                        ? message.getContentBlocks(ToolUseBlock.class) : List.of();
                if (toolUses.isEmpty()) {
                    continue;
                }

                int resultEnd = index + 1;
                List<Msg> resultMessages = new ArrayList<>();
                while (resultEnd < messages.size() && messages.get(resultEnd).getRole() == MsgRole.TOOL) {
                    resultMessages.add(messages.get(resultEnd++));
                }
                if (resultMessages.isEmpty()) {
                    continue;
                }

                List<ToolResultBlock> remainingResults = new ArrayList<>();
                for (Msg resultMessage : resultMessages) {
                    remainingResults.addAll(resultMessage.getContentBlocks(ToolResultBlock.class));
                }
                List<ToolResultBlock> orderedResults = new ArrayList<>(remainingResults.size());
                for (ToolUseBlock toolUse : toolUses) {
                    for (int resultIndex = 0; resultIndex < remainingResults.size();) {
                        ToolResultBlock result = remainingResults.get(resultIndex);
                        if (Objects.equals(toolUse.getId(), result.getId())) {
                            orderedResults.add(result);
                            remainingResults.remove(resultIndex);
                        } else {
                            resultIndex++;
                        }
                    }
                }
                orderedResults.addAll(remainingResults);

                int orderedResultIndex = 0;
                for (Msg resultMessage : resultMessages) {
                    List<ContentBlock> reorderedContent = new ArrayList<>(resultMessage.getContent().size());
                    for (ContentBlock block : resultMessage.getContent()) {
                        if (block instanceof ToolResultBlock) {
                            reorderedContent.add(orderedResults.get(orderedResultIndex++));
                        } else {
                            reorderedContent.add(block);
                        }
                    }
                    ordered.add(resultMessage.withContent(reorderedContent));
                }
                index = resultEnd - 1;
            }
            return ordered;
        }

        private List<Content> mergeToolResponseTurns(List<Content> contents) {
            List<Content> merged = new ArrayList<>(contents.size());
            List<Part> pendingToolResponses = new ArrayList<>();
            for (Content content : contents) {
                List<Part> parts = content.parts().orElse(List.of());
                boolean toolResponseTurn = "user".equals(content.role().orElse(null))
                        && !parts.isEmpty()
                        && parts.stream().allMatch(part -> part.functionResponse().isPresent());
                if (toolResponseTurn) {
                    pendingToolResponses.addAll(parts);
                    continue;
                }
                flushToolResponses(merged, pendingToolResponses);
                merged.add(content);
            }
            flushToolResponses(merged, pendingToolResponses);
            return merged;
        }

        private void flushToolResponses(List<Content> contents, List<Part> toolResponses) {
            if (toolResponses.isEmpty()) {
                return;
            }
            contents.add(Content.builder().role("user").parts(List.copyOf(toolResponses)).build());
            toolResponses.clear();
        }
    }

    @Override
    public List<RemoteModelVO> listRemoteModels(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "Gemini");
        String apiBaseUrl = resolveGeminiApiBaseUrl(context.getBaseUrl());
        String url = joinUrl(apiBaseUrl, "/v1beta/models?pageSize=1000");
        log.info("[GeminiAiProvider] 获取 Gemini 远程模型列表: {}", url);

        String response = executeGet(url, Map.of("x-goog-api-key", context.getApiKey()), context.getApiConfig());
        return parseGeminiModels(response);
    }

    private String resolveGeminiApiBaseUrl(String baseUrl) {
        return StrUtil.isBlank(baseUrl)
                ? "https://generativelanguage.googleapis.com"
                : normalizeBaseUrl(baseUrl);
    }
}
