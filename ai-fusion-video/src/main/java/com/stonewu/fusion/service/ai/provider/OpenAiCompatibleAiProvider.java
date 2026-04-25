package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.Model;
import io.agentscope.core.model.OpenAIChatModel;
import org.springframework.ai.chat.model.ChatModel;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * OpenAI 兼容提供商。
 */
@Component
@Slf4j
public class OpenAiCompatibleAiProvider extends AbstractAiProvider {

    private static final Set<String> SUPPORTED_PLATFORMS = Set.of(
            "openai_compatible", "openai", "deepseek", "zhipu", "moonshot", "volcengine", "siliconflow");

    @Override
    public boolean supports(String platform) {
        return platform != null && SUPPORTED_PLATFORMS.contains(platform.toLowerCase());
    }

    @Override
    public ChatModel createChatModel(AiProviderContext context) {
        String platform = context.getPlatform();
        String apiKey = context.getApiKey();
        String baseUrl = resolveRootBaseUrl(platform, context.getBaseUrl());
        String completionsPath = resolveCompletionsPath(context);
        String embeddingsPath = resolveEmbeddingsPath(context);
        Map<String, Object> config = context.getConfig();
        String modelName = context.getModelName();

        requireApiKey(apiKey, "OpenAI Compatible (" + platform + ")");

        OpenAiApi.Builder apiBuilder = OpenAiApi.builder().apiKey(apiKey);
        apiBuilder.restClientBuilder(AiProxySupport.restClientBuilder(
            context.getApiConfig(), 60 * 1000, 3 * 60 * 1000));
        apiBuilder.webClientBuilder(AiProxySupport.webClientBuilder(
            context.getApiConfig(), "openai-compatible-provider", Duration.ofSeconds(60)));
        if (StrUtil.isNotBlank(baseUrl)) {
            apiBuilder.baseUrl(baseUrl);
        }
        apiBuilder.completionsPath(completionsPath);
        apiBuilder.embeddingsPath(embeddingsPath);

        OpenAiChatOptions.Builder optionsBuilder = OpenAiChatOptions.builder().model(modelName);
        applyDouble(config, "temperature", optionsBuilder::temperature);
        applyDouble(config, "topP", optionsBuilder::topP);
        applyInt(config, "maxTokens", optionsBuilder::maxTokens);

        return OpenAiChatModel.builder()
                .openAiApi(apiBuilder.build())
                .defaultOptions(optionsBuilder.build())
                .build();
    }

    @Override
    public Model createAgentScopeModel(AiProviderContext context) {
        String platform = context.getPlatform();
        String apiKey = context.getApiKey();
        String baseUrl = resolveRootBaseUrl(platform, context.getBaseUrl());
        String endpointPath = resolveCompletionsPath(context);

        requireApiKey(apiKey, "OpenAI Compatible (" + platform + ")");

        OpenAIChatModel.Builder builder = OpenAIChatModel.builder()
                .apiKey(apiKey)
                .modelName(context.getModelName())
                .stream(true);
        GenerateOptions generateOptions = buildReasoningOptions(context);
        if (generateOptions != null) {
            builder.generateOptions(generateOptions);
        }
        if (StrUtil.isNotBlank(baseUrl)) {
            builder.baseUrl(baseUrl);
        }
        builder.endpointPath(endpointPath);
        io.agentscope.core.model.transport.HttpTransport proxyTransport =
                AiProxySupport.agentScopeHttpTransport(context.getApiConfig());
        if (proxyTransport != null) {
            builder.httpTransport(proxyTransport);
        }
        return builder.build();
    }

    @Override
    public List<RemoteModelVO> listRemoteModels(AiProviderContext context) {
        String rootBaseUrl = resolveRootBaseUrl(context.getPlatform(), context.getBaseUrl());
        String url = joinUrl(rootBaseUrl, resolveModelsPath(context));

        log.info("[OpenAiCompatibleAiProvider] 获取远程模型列表: {}", url);
        String response = executeGet(url, context.getApiKey() == null
                ? Map.of()
            : Map.of("Authorization", "Bearer " + context.getApiKey()), context.getApiConfig());
        return parseDataArrayModels(response, context.getPlatform());
    }

    private GenerateOptions buildReasoningOptions(AiProviderContext context) {
        GenerateOptions.Builder builder = GenerateOptions.builder();
        boolean hasOptions = false;

        String reasoningEffort = getConfigString(context.getConfig(), "reasoningEffort", "reasoning_effort");
        if (StrUtil.isNotBlank(reasoningEffort)) {
            builder.reasoningEffort(reasoningEffort);
            hasOptions = true;
        }

        Integer thinkingBudget = getConfigInteger(context.getConfig(), "thinkingBudget", "thinking_budget");
        if (thinkingBudget != null) {
            builder.thinkingBudget(thinkingBudget);
            hasOptions = true;
        }

        Boolean includeReasoning = getConfigBoolean(context.getConfig(), "includeReasoning", "include_reasoning");
        if (includeReasoning == null && isReasoningEnabled(context)) {
            includeReasoning = true;
        }
        if (includeReasoning != null) {
            builder.additionalBodyParam("include_reasoning", includeReasoning);
            hasOptions = true;
        }

        return hasOptions ? builder.build() : null;
    }

    private String resolveCompletionsPath(AiProviderContext context) {
        return switch (context.getPlatform().toLowerCase()) {
            case "zhipu" -> "/api/paas/v4/chat/completions";
            case "volcengine" -> "/api/v3/chat/completions";
            default -> shouldAutoAppendV1Path(context) ? "/v1/chat/completions" : "/chat/completions";
        };
    }

    private String resolveEmbeddingsPath(AiProviderContext context) {
        return switch (context.getPlatform().toLowerCase()) {
            case "zhipu" -> "/api/paas/v4/embeddings";
            case "volcengine" -> "/api/v3/embeddings";
            default -> shouldAutoAppendV1Path(context) ? "/v1/embeddings" : "/embeddings";
        };
    }

    private String resolveModelsPath(AiProviderContext context) {
        return switch (context.getPlatform().toLowerCase()) {
            case "zhipu" -> "/api/paas/v4/models";
            case "volcengine" -> "/api/v3/models";
            default -> shouldAutoAppendV1Path(context) ? "/v1/models" : "/models";
        };
    }

    private String resolveRootBaseUrl(String platform, String baseUrl) {
        return StrUtil.isBlank(baseUrl) ? inferRootBaseUrl(platform) : normalizeBaseUrl(baseUrl);
    }

    private boolean shouldAutoAppendV1Path(AiProviderContext context) {
        if (!"openai_compatible".equalsIgnoreCase(context.getPlatform())) {
            return true;
        }
        ApiConfig apiConfig = context.getApiConfig();
        return apiConfig == null || !Boolean.FALSE.equals(apiConfig.getAutoAppendV1Path());
    }

    private String inferRootBaseUrl(String platform) {
        return switch (platform.toLowerCase()) {
            case "deepseek" -> "https://api.deepseek.com";
            case "zhipu" -> "https://open.bigmodel.cn";
            case "volcengine" -> "https://ark.cn-beijing.volces.com";
            case "moonshot" -> "https://api.moonshot.cn";
            case "siliconflow" -> "https://api.siliconflow.cn";
            case "openai" -> "https://api.openai.com";
            default -> "https://api.openai.com";
        };
    }
}