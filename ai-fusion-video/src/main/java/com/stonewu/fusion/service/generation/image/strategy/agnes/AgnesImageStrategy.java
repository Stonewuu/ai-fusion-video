package com.stonewu.fusion.service.generation.image.strategy.agnes;

import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.support.AbstractOpenAiCompatibleImageStrategy;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolSupport;
import com.stonewu.fusion.service.media.GenerationMediaService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * Agnes Image 2.0/2.1 strategy using the Agnes generations contract.
 */
@Service
public class AgnesImageStrategy extends AbstractOpenAiCompatibleImageStrategy {

    public static final String PROTOCOL = "agnes";

    @Autowired
    public AgnesImageStrategy(ImageGenerationService imageGenerationService,
                              AiModelService aiModelService,
                              OpenAiCompatibleImageProtocolSupport protocolSupport,
                              AgnesImageProtocolAdapter protocolAdapter,
                              GenerationMediaService generationMediaService) {
        super(imageGenerationService, aiModelService, protocolSupport, protocolAdapter, generationMediaService);
    }

    public AgnesImageStrategy(ImageGenerationService imageGenerationService,
                              AiModelService aiModelService,
                              ModelPresetService modelPresetService,
                              GenerationMediaService generationMediaService,
                              StorageConfigService storageConfigService,
                              PresetArtStyleResourceResolver presetArtStyleResourceResolver) {
        this(imageGenerationService, aiModelService,
                createWiring(modelPresetService, storageConfigService,
                        presetArtStyleResourceResolver), generationMediaService);
    }

    private AgnesImageStrategy(ImageGenerationService imageGenerationService,
                               AiModelService aiModelService,
                               Wiring wiring,
                               GenerationMediaService generationMediaService) {
        super(imageGenerationService, aiModelService, wiring.support(), wiring.adapter(), generationMediaService);
    }

    @Override
    public String getName() {
        return PROTOCOL;
    }

    @Override
    protected String defaultModelCode() {
        return "agnes-image-2.1-flash";
    }

    private static Wiring createWiring(ModelPresetService modelPresetService,
                                       StorageConfigService storageConfigService,
                                       PresetArtStyleResourceResolver presetArtStyleResourceResolver) {
        OpenAiCompatibleImageProtocolSupport support = new OpenAiCompatibleImageProtocolSupport(
                modelPresetService, storageConfigService, presetArtStyleResourceResolver);
        return new Wiring(support, new AgnesImageProtocolAdapter(support));
    }

    private record Wiring(OpenAiCompatibleImageProtocolSupport support,
                          AgnesImageProtocolAdapter adapter) {
    }
}
