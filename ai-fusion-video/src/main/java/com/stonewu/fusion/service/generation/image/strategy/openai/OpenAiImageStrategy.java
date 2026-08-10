package com.stonewu.fusion.service.generation.image.strategy.openai;

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

/** Strict OpenAI Images API strategy (generations and edits only). */
@Service
public class OpenAiImageStrategy extends AbstractOpenAiCompatibleImageStrategy {

    public static final String PROTOCOL = "openai";

    @Autowired
    public OpenAiImageStrategy(ImageGenerationService imageGenerationService,
                               AiModelService aiModelService,
                               OpenAiCompatibleImageProtocolSupport protocolSupport,
                               OpenAiImageProtocolAdapter protocolAdapter) {
        super(imageGenerationService, aiModelService, protocolSupport, protocolAdapter);
    }

    public OpenAiImageStrategy(ImageGenerationService imageGenerationService,
                               AiModelService aiModelService,
                               ModelPresetService modelPresetService,
                               MediaStorageService mediaStorageService,
                               StorageConfigService storageConfigService,
                               PresetArtStyleResourceResolver presetArtStyleResourceResolver) {
        this(imageGenerationService, aiModelService,
                createWiring(modelPresetService, mediaStorageService, storageConfigService,
                        presetArtStyleResourceResolver));
    }

    private OpenAiImageStrategy(ImageGenerationService imageGenerationService,
                                AiModelService aiModelService,
                                Wiring wiring) {
        super(imageGenerationService, aiModelService, wiring.support(), wiring.adapter());
    }

    @Override
    public String getName() {
        return PROTOCOL;
    }

    private static Wiring createWiring(ModelPresetService modelPresetService,
                                       MediaStorageService mediaStorageService,
                                       StorageConfigService storageConfigService,
                                       PresetArtStyleResourceResolver presetArtStyleResourceResolver) {
        OpenAiCompatibleImageProtocolSupport support = new OpenAiCompatibleImageProtocolSupport(
                modelPresetService, mediaStorageService, storageConfigService, presetArtStyleResourceResolver);
        return new Wiring(support, new OpenAiImageProtocolAdapter(support));
    }

    private record Wiring(OpenAiCompatibleImageProtocolSupport support,
                          OpenAiImageProtocolAdapter adapter) {
    }
}
