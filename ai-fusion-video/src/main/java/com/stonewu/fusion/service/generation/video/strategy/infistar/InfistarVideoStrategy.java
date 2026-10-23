package com.stonewu.fusion.service.generation.video.strategy.infistar;

import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.support.AbstractOpenAiCompatibleVideoStrategy;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import org.springframework.stereotype.Component;

@Component
public class InfistarVideoStrategy extends AbstractOpenAiCompatibleVideoStrategy {
    public InfistarVideoStrategy(AiModelService models, ApiConfigService configs, VideoGenerationService videos,
                                 ModelPresetService presets, AiModelMetadataResolver metadata,
                                 OpenAiCompatibleVideoProtocolSupport support, InfistarVideoProtocolAdapter adapter) {
        super(models, configs, videos, presets, metadata, support, adapter);
    }

    @Override
    public String getName() {
        return "infistar_video";
    }

    @Override
    protected String providerLabel() {
        return "无限星河";
    }
}
