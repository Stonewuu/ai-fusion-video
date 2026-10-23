package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.OutputConfig;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.transport.ProxyConfig;
import io.agentscope.extensions.model.anthropic.AnthropicChatModel;
import io.agentscope.extensions.model.anthropic.formatter.AnthropicChatFormatter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.anthropic.AnthropicChatOptions;
import org.springframework.ai.anthropic.api.AnthropicApi;
import org.springframework.ai.chat.model.ChatModel;
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
        var rest = AiProxySupport.restClientBuilder(context.getApiConfig(), 60 * 1000, 25 * 60 * 1000);
        var web = AiProxySupport.webClientBuilder(context.getApiConfig(), "anthropic-provider", Duration.ofMinutes(25));
        if (GatewayAuthentication.bearer(context.getConfig())) {
            rest.defaultHeader("Authorization", "Bearer " + context.getApiKey());
            web.defaultHeader("Authorization", "Bearer " + context.getApiKey());
        }
        apiBuilder.restClientBuilder(rest);
        apiBuilder.webClientBuilder(web);
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
    public ChatModelBase createAgentScopeModel(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "Anthropic");
        GenerateOptions defaultOptions = buildReasoningOptions(context);
        String rootBaseUrl = resolveRootBaseUrl(context.getBaseUrl());
        if (context.isSingleRequest()) {
            return new SingleRequestAnthropicModel(context, rootBaseUrl,
                    GenerateOptions.mergeOptions(defaultOptions, buildGeminiGenerateOptions(context)));
        }
        AnthropicChatModel.Builder builder = AnthropicChatModel.builder()
                .modelName(context.getModelName())
                .stream(true)
                .formatter(agentScopeFormatter())
                .baseUrl(rootBaseUrl);
        if (GatewayAuthentication.bearer(context.getConfig())) {
            builder.authToken(context.getApiKey());
        } else {
            builder.apiKey(context.getApiKey());
        }
        if (defaultOptions != null) {
            builder.defaultOptions(defaultOptions);
        }
        ProxyConfig proxy = AiProxySupport.agentScopeProxyConfig(context.getApiConfig());
        if (proxy != null) {
            builder.proxy(proxy);
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

        String reasoningEffort = getConfigString(
                context.getConfig(), "reasoningEffort", "reasoning_effort");
        if (StrUtil.isNotBlank(reasoningEffort)) {
            builder.reasoningEffort(reasoningEffort);
            hasOptions = true;
        }

        return hasOptions ? builder.build() : null;
    }

    static AnthropicChatFormatter agentScopeFormatter() {
        return new ReasoningEffortAnthropicFormatter();
    }

    private static final class ReasoningEffortAnthropicFormatter extends AnthropicChatFormatter {

        @Override
        public void applyOptions(
                MessageCreateParams.Builder paramsBuilder,
                GenerateOptions options,
                GenerateOptions defaultOptions) {
            super.applyOptions(paramsBuilder, options, defaultOptions);
            GenerateOptions merged = GenerateOptions.mergeOptions(options, defaultOptions);
            if (merged == null || StrUtil.isBlank(merged.getReasoningEffort())) {
                return;
            }
            paramsBuilder.outputConfig(OutputConfig.builder()
                    .effort(OutputConfig.Effort.of(merged.getReasoningEffort().trim()))
                    .build());
        }
    }

    private String resolveRootBaseUrl(String baseUrl) {
        return StrUtil.isBlank(baseUrl) ? "https://api.anthropic.com" : normalizeBaseUrl(baseUrl).replaceAll("/v1$", "");
    }
}
