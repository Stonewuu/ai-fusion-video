package com.stonewu.fusion.service.generation.video.strategy;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Routes video generation by the explicitly configured effective request protocol. */
@Component
@RequiredArgsConstructor
public class VideoGenerationStrategyRouter {

    private final List<VideoGenerationStrategy> strategies;
    private final AiModelMetadataResolver aiModelMetadataResolver;

    private volatile Map<String, VideoGenerationStrategy> strategyMap;

    public VideoGenerationStrategy resolve(AiModel model) {
        if (model == null) {
            throw new BusinessException("视频模型不存在");
        }

        Map<String, VideoGenerationStrategy> candidates = getStrategyMap();
        if (candidates.isEmpty()) {
            throw new BusinessException("没有可用的视频生成策略");
        }

        AiModelMetadata metadata = aiModelMetadataResolver.resolve(model);
        String protocol = normalizeStrategyName(metadata.modelProtocol());
        if (StrUtil.isBlank(protocol)) {
            throw new BusinessException("视频模型未配置请求协议：请在模型中设置覆盖协议，或在 API 配置中设置视频默认协议");
        }

        VideoGenerationStrategy strategy = candidates.get(protocol);
        if (strategy != null) {
            return strategy;
        }

        throw new BusinessException("未找到匹配的视频生成策略: protocol=" + protocol
                + ", platform=" + metadata.platform());
    }

    private Map<String, VideoGenerationStrategy> getStrategyMap() {
        if (strategyMap == null) {
            synchronized (this) {
                if (strategyMap == null) {
                    Map<String, VideoGenerationStrategy> map = new LinkedHashMap<>();
                    for (VideoGenerationStrategy strategy : strategies) {
                        for (String protocol : strategy.getSupportedProtocols()) {
                            map.putIfAbsent(normalizeStrategyName(protocol), strategy);
                        }
                    }
                    strategyMap = map;
                }
            }
        }
        return strategyMap;
    }

    private String normalizeStrategyName(String name) {
        String normalized = aiModelMetadataResolver.normalizeProtocol(name);
        return StrUtil.isBlank(normalized) ? "" : normalized;
    }
}
