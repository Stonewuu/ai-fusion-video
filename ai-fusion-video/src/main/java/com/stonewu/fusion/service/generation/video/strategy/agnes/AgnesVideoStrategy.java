package com.stonewu.fusion.service.generation.video.strategy.agnes;

import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.support.AbstractOpenAiCompatibleVideoStrategy;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import org.springframework.stereotype.Component;

/**
 * Agnes video protocol strategy, independent from the OpenAI Videos contract.
 */
@Component
public class AgnesVideoStrategy extends AbstractOpenAiCompatibleVideoStrategy {

    public static final String PROTOCOL = "agnes";

    public AgnesVideoStrategy(AiModelService aiModelService,
                              ApiConfigService apiConfigService,
                              VideoGenerationService videoGenerationService,
                              ModelPresetService modelPresetService,
                              AiModelMetadataResolver aiModelMetadataResolver,
                              OpenAiCompatibleVideoProtocolSupport protocolSupport,
                              AgnesVideoProtocolAdapter protocolAdapter) {
        super(aiModelService, apiConfigService, videoGenerationService, modelPresetService,
                aiModelMetadataResolver, protocolSupport, protocolAdapter);
    }

    @Override
    public String getName() {
        return PROTOCOL;
    }
}
