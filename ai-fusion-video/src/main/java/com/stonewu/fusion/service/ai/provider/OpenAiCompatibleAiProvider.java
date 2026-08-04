package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.transport.HttpTransport;
import io.agentscope.extensions.model.openai.OpenAIChatModel;
import org.springframework.ai.chat.model.ChatModel;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * OpenAI 兼容提供商。
 */
@Component
@Slf4j
public class OpenAiCompatibleAiProvider extends AbstractAiProvider {

    private static final Set<String> SUPPORTED_PLATFORMS = Set.of(
            "openai_compatible", "openai", "agnes", "deepseek", "zhipu", "moonshot", "volcengine", "siliconflow", "newapi");

    @Override
    public boolean supports(String platform) {
        return platform != null && SUPPORTED_PLATFORMS.contains(platform.toLowerCase());
    }

    @Override
    public ChatModel createChatModel(AiProviderContext context) {
        String platform = context.getPlatform();
        String apiKey = context.getApiKey();
        String baseUrl = resolveRootBaseUrl(context);
        String completionsPath = resolveCompletionsPath(context);
        String embeddingsPath = resolveEmbeddingsPath(context);
        Map<String, Object> config = context.getConfig();
        String modelName = context.getModelName();

        requireApiKey(apiKey, "OpenAI Compatible (" + platform + ")");

        if (shouldUseResponsesApi(context)) {
            log.warn("[OpenAiCompatibleAiProvider] Responses API 目前仅接入 AgentScope 主链路，Spring AI ChatModel 仍回退到 chat/completions: model={}",
                    context.getModelName());
        }

        log.info("[OpenAiCompatibleAiProvider] 创建 ChatModel: platform={}, authPlatform={}, baseUrl={}, completionsPath={}, model={}",
                platform,
                context.getApiConfig() != null ? context.getApiConfig().getPlatform() : null,
                baseUrl,
                completionsPath,
                modelName);

        OpenAiApi.Builder apiBuilder = OpenAiApi.builder().apiKey(apiKey);
        apiBuilder.restClientBuilder(AiProxySupport.restClientBuilder(
            context.getApiConfig(), 60 * 1000, 25 * 60 * 1000));
        apiBuilder.webClientBuilder(AiProxySupport.webClientBuilder(
            context.getApiConfig(), "openai-compatible-provider", Duration.ofMinutes(25)));
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
    public ChatModelBase createAgentScopeModel(AiProviderContext context) {
        String platform = context.getPlatform();
        String apiKey = context.getApiKey();
        String baseUrl = resolveRootBaseUrl(context);
        String endpointPath = resolveCompletionsPath(context);

        requireApiKey(apiKey, "OpenAI Compatible (" + platform + ")");

        GenerateOptions generateOptions = buildGenerateOptions(context);
        if (shouldUseResponsesApi(context)) {
            return new OpenAiResponsesAgentScopeModel(
                    context.getApiConfig(),
                    apiKey,
                    baseUrl,
                    context.getModelName(),
                    generateOptions);
        }

        OpenAIChatModel.Builder builder = OpenAIChatModel.builder()
                .apiKey(apiKey)
                .modelName(context.getModelName())
                .stream(true);
        if (generateOptions != null) {
            builder.generateOptions(generateOptions);
        }
        if (StrUtil.isNotBlank(baseUrl)) {
            builder.baseUrl(baseUrl);
        }
        builder.endpointPath(endpointPath);
        HttpTransport proxyTransport = AiProxySupport.agentScopeHttpTransport(context.getApiConfig());
        if (proxyTransport != null) {
            builder.httpTransport(proxyTransport);
        }
        return builder.build();
    }

    @Override
    public List<RemoteModelVO> listRemoteModels(AiProviderContext context) {
        String rootBaseUrl = resolveRootBaseUrl(context);
        String url = joinUrl(rootBaseUrl, resolveModelsPath(context));

        log.info("[OpenAiCompatibleAiProvider] 获取远程模型列表: {}", url);
        String response = executeGet(url, context.getApiKey() == null
                ? Map.of()
            : Map.of("Authorization", "Bearer " + context.getApiKey()), context.getApiConfig());
        return parseDataArrayModels(response, context.getPlatform());
    }

    private GenerateOptions buildGenerateOptions(AiProviderContext context) {
        GenerateOptions.Builder builder = GenerateOptions.builder();
        boolean hasOptions = false;

        Double temperature = getConfigDoubleValue(context.getConfig(), "temperature");
        if (temperature != null) {
            builder.temperature(temperature);
            hasOptions = true;
        }

        Double topP = getConfigDoubleValue(context.getConfig(), "topP", "top_p");
        if (topP != null) {
            builder.topP(topP);
            hasOptions = true;
        }

        Integer maxTokens = getConfigInteger(context.getConfig(), "maxTokens", "max_tokens");
        if (maxTokens != null) {
            builder.maxTokens(maxTokens);
            builder.maxCompletionTokens(maxTokens);
            hasOptions = true;
        }

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
        if (includeReasoning == null && isReasoningEnabled(context) && !usesChatTemplateThinking(context)) {
            includeReasoning = true;
        }
        if (includeReasoning != null) {
            builder.additionalBodyParam("include_reasoning", includeReasoning);
            hasOptions = true;
        }

        // Agnes 等 OpenAI 兼容网关通过 chat_template_kwargs.enable_thinking 开启 Thinking。
        if (applyChatTemplateThinking(builder, context)) {
            hasOptions = true;
        }

        return hasOptions ? builder.build() : null;
    }

    /**
     * Agnes OpenAI-compatible Thinking uses {@code chat_template_kwargs.enable_thinking}.
     * Official OpenAI models should keep using include_reasoning / Responses API instead.
     */
    private boolean applyChatTemplateThinking(GenerateOptions.Builder builder, AiProviderContext context) {
        Object configured = getConfigValue(context.getConfig(),
                "chatTemplateKwargs", "chat_template_kwargs");
        if (configured instanceof Map<?, ?> map) {
            builder.additionalBodyParam("chat_template_kwargs", map);
            return true;
        }

        if (!usesChatTemplateThinking(context)) {
            return false;
        }

        Boolean enableThinking = getConfigBoolean(context.getConfig(),
                "enableThinking", "enable_thinking");
        if (enableThinking == null) {
            if (!isReasoningEnabled(context)) {
                return false;
            }
            enableThinking = true;
        }

        Map<String, Object> kwargs = new LinkedHashMap<>();
        kwargs.put("enable_thinking", enableThinking);
        builder.additionalBodyParam("chat_template_kwargs", kwargs);
        return true;
    }

    private boolean usesChatTemplateThinking(AiProviderContext context) {
        Boolean explicit = getConfigBoolean(context.getConfig(),
                "useChatTemplateThinking", "use_chat_template_thinking", "chatTemplateThinking");
        if (explicit != null) {
            return explicit;
        }
        String modelName = context.getModelName();
        return modelName != null && modelName.toLowerCase(Locale.ROOT).contains("agnes");
    }

    private boolean shouldUseResponsesApi(AiProviderContext context) {
        Boolean useResponsesApi = getConfigBoolean(context.getConfig(),
                "useResponsesApi", "useResponses", "responseApi", "responsesApi");
        if (useResponsesApi != null) {
            return useResponsesApi;
        }

        String apiMode = getConfigString(context.getConfig(),
                "apiMode", "api_mode", "openaiApiMode", "openai_api_mode");
        if (StrUtil.isBlank(apiMode)) {
            return false;
        }

        String normalized = apiMode.trim().toLowerCase();
        return "responses".equals(normalized) || "response".equals(normalized);
    }

    private Double getConfigDoubleValue(Map<String, Object> config, String... keys) {
        Object value = getConfigValue(config, keys);
        if (value == null) {
            return null;
        }
        try {
            return toDouble(value);
        } catch (Exception e) {
            log.warn("[OpenAiCompatibleAiProvider] 参数解析失败: keys={}, value={}", String.join(",", keys), value);
            return null;
        }
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

    /**
     * 解析实际服务根地址。
     * <p>
     * context.platform 可能是请求协议（如 openai_compatible），而不是鉴权平台（如 agnes）。
     * 当 baseUrl 为空时，必须优先按 API 配置的鉴权平台推断默认域名。
     */
    private String resolveRootBaseUrl(AiProviderContext context) {
        if (StrUtil.isNotBlank(context.getBaseUrl())) {
            return normalizeBaseUrl(context.getBaseUrl());
        }
        ApiConfig apiConfig = context.getApiConfig();
        if (apiConfig != null) {
            String fromConfig = ApiConfigService.resolveEffectiveApiUrlStatic(apiConfig);
            if (StrUtil.isNotBlank(fromConfig)) {
                return normalizeBaseUrl(fromConfig);
            }
        }
        return inferRootBaseUrl(context.getPlatform());
    }

    private boolean shouldAutoAppendV1Path(AiProviderContext context) {
        ApiConfig apiConfig = context.getApiConfig();
        String authPlatform = apiConfig != null && StrUtil.isNotBlank(apiConfig.getPlatform())
                ? apiConfig.getPlatform()
                : context.getPlatform();
        if (!"openai_compatible".equalsIgnoreCase(authPlatform) && !"agnes".equalsIgnoreCase(authPlatform)) {
            return true;
        }
        return apiConfig == null || !Boolean.FALSE.equals(apiConfig.getAutoAppendV1Path());
    }

    private String inferRootBaseUrl(String platform) {
        if (StrUtil.isBlank(platform)) {
            return "https://api.openai.com";
        }
        String defaultUrl = ApiConfigService.platformDefaultApiUrl(platform);
        if (StrUtil.isNotBlank(defaultUrl) && defaultUrl.startsWith("http")) {
            return defaultUrl;
        }
        return switch (platform.toLowerCase(Locale.ROOT)) {
            case "deepseek" -> "https://api.deepseek.com";
            case "zhipu" -> "https://open.bigmodel.cn";
            case "moonshot" -> "https://api.moonshot.cn";
            case "siliconflow" -> "https://api.siliconflow.cn";
            default -> "https://api.openai.com";
        };
    }
}
