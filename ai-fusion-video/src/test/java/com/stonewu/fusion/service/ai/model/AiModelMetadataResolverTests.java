package com.stonewu.fusion.service.ai.model;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.ApiConfigService;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.Mockito.mock;

class AiModelMetadataResolverTests {

    private final AiModelMetadataResolver resolver =
            new AiModelMetadataResolver(mock(ApiConfigService.class));

    @Test
    void modelProtocolOverrideWinsProviderDefault() {
        AiModel model = AiModel.builder()
                .modelType(3)
                .modelProtocol("jimeng")
                .build();
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("newapi")
                .videoProtocol("newapi")
                .build();

        AiModelMetadata metadata = resolver.resolve(model, apiConfig);

        assertEquals("jimeng", metadata.modelProtocol());
    }

    @Test
    void inheritsCapabilitySpecificProviderProtocols() {
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai_compatible")
                .textProtocol("openai_compatible")
                .imageProtocol("agnes")
                .videoProtocol("agnes")
                .build();

        assertEquals("openai_compatible", resolver.resolve(
                AiModel.builder().modelType(1).build(), apiConfig).modelProtocol());
        assertEquals("agnes", resolver.resolve(
                AiModel.builder().modelType(2).build(), apiConfig).modelProtocol());
        assertEquals("agnes", resolver.resolve(
                AiModel.builder().modelType(3).build(), apiConfig).modelProtocol());
    }

    @Test
    void doesNotInferProtocolFromModelNameCodeOrPlatform() {
        AiModel model = AiModel.builder()
                .name("Agnes Video")
                .code("agnes-video-v2.0")
                .modelType(3)
                .build();

        AiModelMetadata metadata = resolver.resolve(model,
                ApiConfig.builder().platform("openai_compatible").build());

        assertNull(metadata.modelProtocol());
    }

    @Test
    void remoteDiscoveryInfersOnlyModelType() {
        RemoteModelMetadata metadata = resolver.resolveRemoteModel(
                "openai_compatible", "agnes-video-v2.0", "Agnes Video", null);

        assertEquals(3, metadata.modelType());
        assertNull(metadata.modelProtocol());
        assertEquals("openai_compatible", metadata.providerPlatform());
        assertEquals("Agnes Video", metadata.displayName());
    }
}
