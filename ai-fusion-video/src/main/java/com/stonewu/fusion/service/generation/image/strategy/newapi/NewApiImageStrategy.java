package com.stonewu.fusion.service.generation.image.strategy.newapi;

import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.support.AbstractOpenAiCompatibleImageStrategy;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolSupport;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/** NewAPI/OpenAI-shaped extensions, including asynchronous image tasks. */
@Service
public class NewApiImageStrategy extends AbstractOpenAiCompatibleImageStrategy {

    public static final String PLATFORM = "newapi";

    @Autowired
    public NewApiImageStrategy(ImageGenerationService imageGenerationService,
                               AiModelService aiModelService,
                               OpenAiCompatibleImageProtocolSupport protocolSupport,
                               NewApiImageProtocolAdapter protocolAdapter) {
        super(imageGenerationService, aiModelService, protocolSupport, protocolAdapter);
    }

    public NewApiImageStrategy(ImageGenerationService imageGenerationService,
                               AiModelService aiModelService,
                               ModelPresetService modelPresetService,
                               MediaStorageService mediaStorageService,
                               StorageConfigService storageConfigService,
                               PresetArtStyleResourceResolver presetArtStyleResourceResolver) {
        this(imageGenerationService, aiModelService,
                createWiring(modelPresetService, mediaStorageService, storageConfigService,
                        presetArtStyleResourceResolver));
    }

    private NewApiImageStrategy(ImageGenerationService imageGenerationService,
                                AiModelService aiModelService,
                                Wiring wiring) {
        super(imageGenerationService, aiModelService, wiring.support(), wiring.adapter());
    }

    @Override
    public String getName() {
        return PLATFORM;
    }

    private static Wiring createWiring(ModelPresetService modelPresetService,
                                       MediaStorageService mediaStorageService,
                                       StorageConfigService storageConfigService,
                                       PresetArtStyleResourceResolver presetArtStyleResourceResolver) {
        OpenAiCompatibleImageProtocolSupport support = new OpenAiCompatibleImageProtocolSupport(
                modelPresetService, mediaStorageService, storageConfigService, presetArtStyleResourceResolver);
        return new Wiring(support, new NewApiImageProtocolAdapter(support));
    }

    private record Wiring(OpenAiCompatibleImageProtocolSupport support,
                          NewApiImageProtocolAdapter adapter) {
    }
}
