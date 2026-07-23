package com.stonewu.fusion.service.generation.video.strategy.openai;

import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.support.AbstractOpenAiCompatibleVideoStrategy;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import com.stonewu.fusion.service.storage.MediaStorageService;
import org.springframework.stereotype.Component;

/** Strict implementation of the official OpenAI/Sora Videos API. */
@Component
public class OpenAiVideoStrategy extends AbstractOpenAiCompatibleVideoStrategy {

    public static final String PROTOCOL = "openai";

    public OpenAiVideoStrategy(AiModelService aiModelService,
                               ApiConfigService apiConfigService,
                               VideoGenerationService videoGenerationService,
                               ModelPresetService modelPresetService,
                               MediaStorageService mediaStorageService,
                               AiModelMetadataResolver aiModelMetadataResolver,
                               OpenAiCompatibleVideoProtocolSupport protocolSupport,
                               OpenAiVideoProtocolAdapter protocolAdapter) {
        super(aiModelService, apiConfigService, videoGenerationService, modelPresetService,
                mediaStorageService, aiModelMetadataResolver, protocolSupport, protocolAdapter);
    }

    @Override
    public String getName() {
        return PROTOCOL;
    }
}
