package com.stonewu.fusion.service.ai;

import cn.hutool.json.JSONUtil;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelPresetServiceTests {

    private ModelPresetService service;

    @BeforeEach
    void setUp() {
        service = new ModelPresetService();
        service.init();
    }

    @Test
    void shouldLoadOnlyRecentAndCurrentlyAvailableOpenAiAndGoogleImageGenerations() {
        assertFalse(service.hasPreset("dall-e-3"));
        assertFalse(service.hasPreset("imagen-3.0-generate-002"));
        assertFalse(service.hasPreset("imagen-4.0-generate-001"));
        assertFalse(service.hasPreset("imagen-4.0-fast-generate-001"));
        assertFalse(service.hasPreset("imagen-4.0-ultra-generate-001"));

        List.of(
                "gpt-image-1",
                "gpt-image-1-mini",
                "gpt-image-1.5",
                "gpt-image-2",
                "sora-2",
                "sora-2-pro",
                "gemini-2.5-flash-image",
                "gemini-3.0-pro-image",
                "gemini-3.1-flash-image"
        ).forEach(code -> assertTrue(service.hasPreset(code), code));
    }

    @Test
    void shouldResolveKlingPresetByActualModelCodeAndType() {
        assertEquals("kling-v3-image", service.findPresetCode("kling-v3", 2));
        assertEquals("kling-v3-video", service.findPresetCode("kling-v3", 3));
        assertEquals("kling-v3-omni-image", service.findPresetCode("kling-v3-omni", 2));
        assertEquals("kling-v3-omni-video", service.findPresetCode("kling-v3-omni", 3));

        var imagePreset = service.getPreset("kling-v3-image");
        var videoPreset = service.getPreset("kling-v3-video");
        assertNotNull(imagePreset);
        assertNotNull(videoPreset);
        assertEquals("kling-v3", imagePreset.getStr("modelCode"));
        assertEquals("kling-v3", videoPreset.getStr("modelCode"));
        assertEquals(2, imagePreset.getInt("modelType"));
        assertEquals(3, videoPreset.getInt("modelType"));
    }

    @Test
    void shouldResolveJimengImageAndVideoCapabilityPresets() {
        assertEquals("jimeng-4.0-image", service.findPresetCode("jimeng-4.0", 2));
        assertEquals("jimeng-video-3.0-pro", service.findPresetCode("jimeng-video-3.0-pro", 3));
        assertEquals(List.of("url"), JSONUtil.toList(
                service.getPreset("jimeng-4.0-image").getJSONObject("config")
                        .getJSONArray("referenceImageInputFormats"), String.class));
    }

    @Test
    void shouldDeclareTransportFormatsForEveryPresetWithImageInputs() {
        service.getAllPresets().forEach(preset -> {
            var config = preset.getJSONObject("config");
            int modelType = preset.getInt("modelType", 0);
            boolean hasImageInputs = modelType == 2 && config.getBool("supportReferenceImages", false)
                    || modelType == 3 && (config.getBool("supportFirstFrame", false)
                    || config.getBool("supportLastFrame", false)
                    || config.getBool("supportReferenceImages", false));
            if (hasImageInputs) {
                assertFalse(config.getJSONArray("referenceImageInputFormats").isEmpty(),
                        preset.getStr("code"));
            }
        });
    }

    @Test
    void shouldEnableUrlAndDataUriForGptImagePresetsByDefault() {
        List.of(
                "gpt-image-1",
                "gpt-image-1-mini",
                "gpt-image-1.5",
                "gpt-image-2"
        ).forEach(code -> assertEquals(
                List.of("url", "data_uri"),
                JSONUtil.toList(service.getPreset(code).getJSONObject("config")
                        .getJSONArray("referenceImageInputFormats"), String.class),
                code));
    }

    @Test
    void shouldExposeCorrectedVolcengineAndSoraCapabilities() {
        var seedance = service.getPreset("doubao-seedance-2-0-260128").getJSONObject("config");
        assertEquals(List.of("480p", "720p", "1080p", "4k"),
                JSONUtil.toList(seedance.getJSONArray("supportedResolutions"), String.class));
        assertFalse(seedance.getBool("supportCameraFixed"));
        assertTrue(seedance.getBool("exclusiveInputModes"));

        var seedreamPro = service.getPreset("doubao-seedream-5-0-pro-260628").getJSONObject("config");
        assertEquals(10, seedreamPro.getInt("maxReferenceImages"));
        assertFalse(seedreamPro.getBool("supportSequentialImages"));
        assertEquals("2816x1584",
                seedreamPro.getJSONObject("supportedSizes").getJSONObject("2K").getStr("16:9"));

        var sora2Pro = service.getPreset("sora-2-pro").getJSONObject("config");
        assertEquals(List.of(4, 8, 12, 16, 20),
                JSONUtil.toList(sora2Pro.getJSONArray("supportedDurations"), Integer.class));
        assertTrue(sora2Pro.getBool("supportGenerateAudio"));
        assertTrue(JSONUtil.toList(sora2Pro.getJSONArray("supportedResolutions"), String.class)
                .contains("1920x1080"));

        var gptImage2 = service.getPreset("gpt-image-2").getJSONObject("config");
        assertFalse(gptImage2.containsKey("inputFidelity"));
    }

    @Test
    void shouldExposeCurrentTextModelsWithConsistentMultimodalCapabilities() {
        List.of(
                "gpt-5.6-sol",
                "gpt-5.6-terra",
                "gpt-5.6-luna",
                "gemini-3.6-flash",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "claude-fable-5",
                "claude-opus-5",
                "claude-sonnet-5",
                "claude-haiku-4-5-20251001",
                "qwen3.7-plus",
                "qwen3.5-omni-plus",
                "doubao-seed-2-1-pro-260628",
                "doubao-seed-2-1-turbo-260628",
                "deepseek-v4-flash",
                "deepseek-v4-pro"
        ).forEach(code -> assertTrue(service.hasPreset(code), code));

        assertFalse(service.hasPreset("deepseek-chat"));
        assertFalse(service.hasPreset("deepseek-reasoner"));

        service.getPresetsByType(1).forEach(preset -> {
            var types = preset.getJSONArray("multimodalInputTypes");
            var transports = preset.getJSONObject("multimodalInputTransports");
            Integer contextWindow = preset.getInt("contextWindow");
            var reasoningEffortLevels = preset.getJSONArray("reasoningEffortLevels");
            assertNotNull(types, preset.getStr("code"));
            assertNotNull(transports, preset.getStr("code"));
            assertNotNull(contextWindow, preset.getStr("code"));
            assertNotNull(reasoningEffortLevels, preset.getStr("code"));
            assertTrue(contextWindow > 0, preset.getStr("code"));

            Set<String> enabledTypes = Set.copyOf(JSONUtil.toList(types, String.class));
            assertEquals(enabledTypes, transports.keySet(), preset.getStr("code"));
            assertTrue(Set.of("image", "video", "audio", "file").containsAll(enabledTypes),
                    preset.getStr("code"));
            enabledTypes.forEach(type -> {
                List<String> declaredTransports = JSONUtil.toList(
                        transports.getJSONArray(type), String.class);
                assertFalse(declaredTransports.isEmpty(), preset.getStr("code") + ":" + type);
                assertTrue(Set.of("url", "base64").containsAll(declaredTransports),
                        preset.getStr("code") + ":" + type);
            });
        });

        assertEquals(List.of("max", "xhigh", "high", "medium", "low", "none"),
                JSONUtil.toList(service.getPreset("gpt-5.6-sol")
                        .getJSONArray("reasoningEffortLevels"), String.class));
        assertEquals(List.of("max", "high"),
                JSONUtil.toList(service.getPreset("deepseek-v4-pro")
                        .getJSONArray("reasoningEffortLevels"), String.class));
        assertEquals("openai_compatible",
                service.getPreset("deepseek-v4-pro").getStr("platform"));
        assertEquals("openai_compatible",
                service.getPreset("deepseek-v4-pro").getStr("modelProtocol"));
        assertEquals(List.of("high", "medium", "low", "minimal"),
                JSONUtil.toList(service.getPreset("gemini-3.6-flash")
                        .getJSONArray("reasoningEffortLevels"), String.class));
        assertTrue(service.getPreset("qwen3.7-plus")
                .getJSONArray("reasoningEffortLevels").isEmpty());
    }
}
