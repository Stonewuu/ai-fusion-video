package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.model.AnthropicChatModel;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.Model;
import org.springframework.ai.anthropic.AnthropicChatModel.Builder;
import org.springframework.ai.anthropic.AnthropicChatOptions;
import org.springframework.ai.anthropic.api.AnthropicApi;
import org.springframework.ai.chat.model.ChatModel;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * Anthropic 提供商。
 */
@Component
@Slf4j
public class AnthropicAiProvider extends AbstractAiProvider {

    @Override
    public boolean supports(String platform) {
        return platform != null && "anthropic".equalsIgnoreCase(platform);
    }

    @Override
    public ChatModel createChatModel(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "Anthropic");

        AnthropicApi.Builder apiBuilder = AnthropicApi.builder().apiKey(context.getApiKey());
        apiBuilder.restClientBuilder(AiProxySupport.restClientBuilder(
            context.getApiConfig(), 60 * 1000, 3 * 60 * 1000));
        apiBuilder.webClientBuilder(AiProxySupport.webClientBuilder(
            context.getApiConfig(), "anthropic-provider", Duration.ofSeconds(60)));
        String rootBaseUrl = resolveRootBaseUrl(context.getBaseUrl());
        if (StrUtil.isNotBlank(rootBaseUrl)) {
            apiBuilder.baseUrl(rootBaseUrl);
        }

        AnthropicChatOptions.Builder optionsBuilder = AnthropicChatOptions.builder()
                .model(context.getModelName());
        applyDouble(context.getConfig(), "temperature", optionsBuilder::temperature);
        applyDouble(context.getConfig(), "topP", optionsBuilder::topP);
        applyInt(context.getConfig(), "maxTokens", optionsBuilder::maxTokens);

        return org.springframework.ai.anthropic.AnthropicChatModel.builder()
                .anthropicApi(apiBuilder.build())
                .defaultOptions(optionsBuilder.build())
                .build();
    }

    @Override
    public Model createAgentScopeModel(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "Anthropic");
        AnthropicChatModel.Builder builder = AnthropicChatModel.builder()
                // .defaultOptions(GenerateOptions.builder().additionalHeader("Authorization", "Bearer " + context.getApiKey()).build())
                .apiKey(context.getApiKey())
                .modelName(context.getModelName())
                .stream(true);
        GenerateOptions defaultOptions = buildReasoningOptions(context);
        if (defaultOptions != null) {
            builder.defaultOptions(defaultOptions);
        }
        String rootBaseUrl = resolveRootBaseUrl(context.getBaseUrl());
        if (StrUtil.isNotBlank(rootBaseUrl)) {
            builder.baseUrl(rootBaseUrl);
        }
        return builder.build();
    }

    @Override
    public List<RemoteModelVO> listRemoteModels(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "Anthropic");
        String apiBaseUrl = ensurePathSuffix(resolveRootBaseUrl(context.getBaseUrl()), "/v1");
        String url = joinUrl(apiBaseUrl, "/models");
        log.info("[AnthropicAiProvider] 获取远程模型列表: {}", url);

        String response = executeGet(url, Map.of(
                "x-api-key", context.getApiKey(),
                "anthropic-version", "2023-06-01"
        ), context.getApiConfig());
        return parseDataArrayModels(response, "anthropic");
    }

    private GenerateOptions buildReasoningOptions(AiProviderContext context) {
        GenerateOptions.Builder builder = GenerateOptions.builder();
        boolean hasOptions = false;

        Object thinking = getConfigValue(context.getConfig(), "thinking");
        if (thinking != null) {
            builder.additionalBodyParam("thinking", thinking);
            hasOptions = true;
        } else if (isReasoningEnabled(context)) {
            Integer thinkingBudget = getConfigInteger(context.getConfig(), "thinkingBudget", "thinking_budget");
            int budgetTokens = thinkingBudget != null ? thinkingBudget : 1024;
            builder.thinkingBudget(budgetTokens);
            builder.additionalBodyParam("thinking", Map.of(
                    "type", "enabled",
                    "budget_tokens", budgetTokens));
            hasOptions = true;
        }

        return hasOptions ? builder.build() : null;
    }

    private String resolveRootBaseUrl(String baseUrl) {
        return StrUtil.isBlank(baseUrl) ? "https://api.anthropic.com" : normalizeBaseUrl(baseUrl);
    }
}