package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
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
    private final AiModelMetadataResolver aiModelMetadataResolver;

    public AiProviderContext createForModel(AiModel model) {
        if (model == null) {
            throw new BusinessException("AI 模型不存在");
        }
        ApiConfig apiConfig = resolveApiConfig(model.getApiConfigId());
        Map<String, Object> config = parseConfig(model.getConfig(), model.getId());
        AiModelMetadata metadata = aiModelMetadataResolver.resolve(model, apiConfig);
        String requestProtocol = metadata.modelProtocol();
        if (StrUtil.isBlank(requestProtocol)) {
            throw new BusinessException("文本模型未配置请求协议：请在模型中设置覆盖协议，或在 API 配置中设置文本默认协议");
        }
        return AiProviderContext.builder()
                .model(model)
                .apiConfig(apiConfig)
                .config(config)
                // platform 使用请求协议，用于路由到对应协议实现；鉴权地址必须按 API 配置平台还原
                .platform(normalizePlatform(requestProtocol))
                .apiKey(apiConfig.getApiKey())
                .baseUrl(apiConfigService.resolveEffectiveApiUrl(apiConfig))
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
        if (apiConfig == null || StrUtil.isBlank(apiConfig.getPlatform())) {
            throw new BusinessException("API 配置未设置接入与鉴权类型");
        }
        return AiProviderContext.builder()
                .apiConfig(apiConfig)
                .platform(normalizePlatform(apiConfig.getPlatform()))
                .apiKey(apiConfig.getApiKey())
                .baseUrl(ApiConfigService.resolveEffectiveApiUrlStatic(apiConfig))
                .build();
    }

    private ApiConfig resolveApiConfig(Long apiConfigId) {
        if (apiConfigId == null) {
            throw new BusinessException("AI 模型未绑定 API 配置");
        }
        ApiConfig apiConfig = apiConfigService.getById(apiConfigId);
        if (apiConfig == null) {
            throw new BusinessException(404, "API 配置不存在");
        }
        return apiConfig;
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

    private String resolveModelName(AiModel model, Map<String, Object> config) {
        Object modelName = config.get("modelName");
        return modelName != null ? modelName.toString() : model.getCode();
    }

    private String normalizePlatform(String platform) {
        return "openai".equalsIgnoreCase(platform) ? "openai_compatible" : platform;
    }
}
