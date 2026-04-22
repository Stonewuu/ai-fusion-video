package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * 统一构建提供商上下文，收敛平台、密钥、地址和模型配置解析逻辑。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class AiProviderContextFactory {

    private final ApiConfigService apiConfigService;

    public AiProviderContext createForModel(AiModel model) {
        if (model == null) {
            throw new BusinessException("AI 模型不存在");
        }
        ApiConfig apiConfig = resolveApiConfig(model.getApiConfigId());
        Map<String, Object> config = parseConfig(model.getConfig(), model.getId());
        String baseUrl = resolveBaseUrl(apiConfig, config);
        return AiProviderContext.builder()
                .model(model)
                .apiConfig(apiConfig)
                .config(config)
                .platform(resolvePlatform(apiConfig, baseUrl))
                .apiKey(resolveApiKey(apiConfig, config))
                .baseUrl(baseUrl)
                .modelName(resolveModelName(model, config))
                .build();
    }

    public AiProviderContext createForApiConfig(Long apiConfigId) {
        ApiConfig apiConfig = apiConfigService.getById(apiConfigId);
        if (apiConfig == null) {
            throw new BusinessException(404, "API 配置不存在");
        }
        return createForApiConfig(apiConfig);
    }

    public AiProviderContext createForApiConfig(ApiConfig apiConfig) {
        String baseUrl = apiConfig != null ? apiConfig.getApiUrl() : null;
        return AiProviderContext.builder()
                .apiConfig(apiConfig)
                .platform(resolvePlatform(apiConfig, baseUrl))
                .apiKey(apiConfig != null ? apiConfig.getApiKey() : null)
                .baseUrl(baseUrl)
                .build();
    }

    private ApiConfig resolveApiConfig(Long apiConfigId) {
        if (apiConfigId == null) {
            return null;
        }
        try {
            return apiConfigService.getById(apiConfigId);
        } catch (Exception e) {
            log.warn("[AiProviderContextFactory] ApiConfig 获取失败: {}", apiConfigId);
            return null;
        }
    }

    private Map<String, Object> parseConfig(String json, Long modelId) {
        if (StrUtil.isBlank(json)) {
            return Map.of();
        }
        try {
            return JSONUtil.parseObj(json);
        } catch (Exception e) {
            log.warn("[AiProviderContextFactory] 配置 JSON 解析失败: modelId={}", modelId, e);
            return Map.of();
        }
    }

    private String resolveApiKey(ApiConfig apiConfig, Map<String, Object> config) {
        if (apiConfig != null && StrUtil.isNotBlank(apiConfig.getApiKey())) {
            return apiConfig.getApiKey();
        }
        Object apiKey = config.get("apiKey");
        return apiKey != null ? apiKey.toString() : null;
    }

    private String resolveBaseUrl(ApiConfig apiConfig, Map<String, Object> config) {
        if (apiConfig != null && StrUtil.isNotBlank(apiConfig.getApiUrl())) {
            return apiConfig.getApiUrl();
        }
        Object baseUrl = config.get("baseUrl");
        return baseUrl != null ? baseUrl.toString() : null;
    }

    private String resolveModelName(AiModel model, Map<String, Object> config) {
        Object modelName = config.get("modelName");
        return modelName != null ? modelName.toString() : model.getCode();
    }

    private String resolvePlatform(ApiConfig apiConfig, String baseUrl) {
        if (apiConfig != null && StrUtil.isNotBlank(apiConfig.getPlatform())) {
            return apiConfig.getPlatform();
        }
        if (StrUtil.isNotBlank(baseUrl)) {
            String url = baseUrl.toLowerCase();
            if (url.contains("deepseek")) return "deepseek";
            if (url.contains("dashscope") || url.contains("aliyuncs")) return "dashscope";
            if (url.contains("bigmodel.cn")) return "zhipu";
            if (url.contains("volces.com") || url.contains("volcengine")) return "volcengine";
            if (url.contains("moonshot")) return "moonshot";
            if (url.contains("siliconflow")) return "siliconflow";
            if (url.contains("anthropic")) return "anthropic";
            if (url.contains("flow2api")) return "GoogleFlowReverseApi";
            if (url.contains("localhost") || url.contains("127.0.0.1")) return "ollama";
            if (url.contains("ai.google.dev") || url.contains("generativelanguage")) return "gemini";
            if (url.contains("googleapis.com") || url.contains("vertex")) return "vertex_ai";
            if (url.contains("openai.com")) return "openai";
            return "openai_compatible";
        }
        return "openai_compatible";
    }
}