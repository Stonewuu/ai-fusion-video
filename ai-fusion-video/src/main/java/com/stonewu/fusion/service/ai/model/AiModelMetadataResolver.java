package com.stonewu.fusion.service.ai.model;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.Locale;

/**
 * 统一解析模型的显式接入与请求协议元数据。
 */
@Service
@RequiredArgsConstructor
public class AiModelMetadataResolver {

    private final ApiConfigService apiConfigService;

    public AiModelMetadata resolve(AiModel model) {
        return resolve(model, resolveApiConfig(model));
    }

    /**
     * 解析模型的显式路由元数据。
     * <p>
     * 请求协议只有两种来源：模型覆盖、API 配置中对应能力类型的默认协议。
     * 模型名称、家族、平台和自定义 JSON 都不参与协议猜测。
     */
    public AiModelMetadata resolve(AiModel model, ApiConfig apiConfig) {
        String platform = apiConfig != null ? apiConfig.getPlatform() : null;
        String protocol = normalizeProtocol(model != null ? model.getModelProtocol() : null);
        if (StrUtil.isBlank(protocol)) {
            protocol = normalizeProtocol(resolveProviderDefaultProtocol(
                    apiConfig, model != null ? model.getModelType() : null));
        }
        return new AiModelMetadata(platform, normalizePlatform(platform), protocol);
    }

    /**
     * 兼容仅掌握平台名称的能力分析调用；该重载不会根据平台推断协议。
     */
    public AiModelMetadata resolve(AiModel model, String platform) {
        ApiConfig apiConfig = resolveApiConfig(model);
        if (apiConfig != null) {
            return resolve(model, apiConfig);
        }
        return new AiModelMetadata(
                platform,
                normalizePlatform(platform),
                normalizeProtocol(model != null ? model.getModelProtocol() : null));
    }

    public RemoteModelMetadata resolveRemoteModel(String providerPlatform, String modelId, String ownedBy, Integer modelType) {
        String normalizedPlatform = normalizePlatform(providerPlatform);
        Integer inferredModelType = inferRemoteModelType(normalizedPlatform, modelId, ownedBy, modelType);
        String displayName = StrUtil.blankToDefault(ownedBy, modelId);
        boolean inferred = modelType == null && inferredModelType != null;
        return new RemoteModelMetadata(normalizedPlatform, displayName, null, inferredModelType, inferred);
    }

    public String resolvePlatform(AiModel model) {
        ApiConfig apiConfig = resolveApiConfig(model);
        return apiConfig != null ? apiConfig.getPlatform() : null;
    }

    public ApiConfig resolveApiConfig(AiModel model) {
        if (model == null || model.getApiConfigId() == null) {
            return null;
        }
        return apiConfigService.getById(model.getApiConfigId());
    }

    public String normalizePlatform(String platform) {
        if (StrUtil.isBlank(platform)) {
            return "";
        }
        String normalized = platform.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "openai" -> "openai_compatible";
            case "vertexai" -> "vertex_ai";
            default -> normalized;
        };
    }

    public String normalizeProtocol(String protocol) {
        if (StrUtil.isBlank(protocol)) {
            return null;
        }
        return protocol.trim().toLowerCase(Locale.ROOT).replace(' ', '_').replace('-', '_');
    }

    private Integer inferRemoteModelType(String platform, String modelId, String ownedBy, Integer currentType) {
        if (currentType != null) {
            return currentType;
        }
        String corpus = corpus(modelId, ownedBy);
        if (containsAny(corpus, "t2v", "i2v", "r2v", "video", "sora", "jimeng", "即梦", "kling", "可灵",
                "seedance", "veo", "pixverse", "hailuo", "wan2.7")) {
            return 3;
        }
        if (containsAny(corpus, "image", "img", "flux", "dall", "imagen", "qwen-image", "wan2.7-image")) {
            return 2;
        }
        return currentType;
    }

    private String resolveProviderDefaultProtocol(ApiConfig apiConfig, Integer modelType) {
        if (apiConfig == null || modelType == null) {
            return null;
        }
        return switch (modelType) {
            case 1 -> apiConfig.getTextProtocol();
            case 2 -> apiConfig.getImageProtocol();
            case 3 -> apiConfig.getVideoProtocol();
            default -> null;
        };
    }

    private String corpus(String primaryText, String secondaryText) {
        return (StrUtil.blankToDefault(primaryText, "") + " " + StrUtil.blankToDefault(secondaryText, ""))
                .toLowerCase(Locale.ROOT);
    }

    private boolean containsAny(String text, String... candidates) {
        if (StrUtil.isBlank(text)) {
            return false;
        }
        for (String candidate : candidates) {
            if (text.contains(candidate.toLowerCase(Locale.ROOT))) {
                return true;
            }
        }
        return false;
    }

}
