package com.stonewu.fusion.service.generation.video.strategy.agnes;

import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.support.AbstractOpenAiCompatibleVideoStrategy;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import org.springframework.stereotype.Component;

@Component
public class Agnes25VideoStrategy extends AbstractOpenAiCompatibleVideoStrategy {
    public static final String PROTOCOL = "agnes_25";

    public Agnes25VideoStrategy(AiModelService models, ApiConfigService configs,
                                VideoGenerationService generation, ModelPresetService presets,
                                AiModelMetadataResolver metadata, OpenAiCompatibleVideoProtocolSupport support,
                                Agnes25VideoProtocolAdapter adapter) {
        super(models, configs, generation, presets, metadata, support, adapter);
    }

    @Override
    public String getName() {
        return PROTOCOL;
    }

    @Override
    protected String providerLabel() {
        return "Agnes 2.5";
    }
}
