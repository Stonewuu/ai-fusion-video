package com.stonewu.fusion.service.generation.image.strategy;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Routes image generation by the explicitly configured effective request protocol. */
@Component
@RequiredArgsConstructor
public class ImageGenerationStrategyRouter {

    private final List<ImageGenerationStrategy> strategies;
    private final AiModelMetadataResolver aiModelMetadataResolver;

    private volatile Map<String, ImageGenerationStrategy> strategyMap;

    public ImageGenerationStrategy resolve(AiModel model, ApiConfig apiConfig) {
        if (model == null) {
            throw new BusinessException("图片模型不存在");
        }
        Map<String, ImageGenerationStrategy> candidates = getStrategyMap();
        if (candidates.isEmpty()) {
            throw new BusinessException("没有可用的图片生成策略");
        }

        AiModelMetadata metadata = aiModelMetadataResolver.resolve(model, apiConfig);
        String protocol = normalize(metadata.modelProtocol());
        if (StrUtil.isBlank(protocol)) {
            throw new BusinessException("图片模型未配置请求协议：请在模型中设置覆盖协议，或在 API 配置中设置图片默认协议");
        }

        ImageGenerationStrategy strategy = candidates.get(protocol);
        if (strategy != null) {
            return strategy;
        }

        throw new BusinessException("未找到匹配的图片生成策略: protocol=" + protocol
                + ", platform=" + metadata.platform());
    }

    private Map<String, ImageGenerationStrategy> getStrategyMap() {
        if (strategyMap == null) {
            synchronized (this) {
                if (strategyMap == null) {
                    Map<String, ImageGenerationStrategy> resolved = new LinkedHashMap<>();
                    for (ImageGenerationStrategy strategy : strategies) {
                        resolved.putIfAbsent(normalize(strategy.getName()), strategy);
                    }
                    strategyMap = resolved;
                }
            }
        }
        return strategyMap;
    }

    private String normalize(String value) {
        String normalized = aiModelMetadataResolver.normalizeProtocol(value);
        return StrUtil.blankToDefault(normalized, "")
                .replace('-', '_').replace(' ', '_');
    }
}
