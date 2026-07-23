package com.stonewu.fusion.service.ai;

import cn.hutool.json.JSONUtil;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

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
}
