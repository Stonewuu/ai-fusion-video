package com.stonewu.fusion.service.generation;

import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.ModelPresetService;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GenerationModelCapabilityServiceTests {

        private final GenerationModelCapabilityService service = new GenerationModelCapabilityService(null, new ModelPresetService());

    @Test
    void shouldMarkOpenAiImageModelAsNoReferenceImageSupport() {
        AiModel model = AiModel.builder()
                .name("GPT Image 1")
                .code("gpt-image-1")
                .build();

        GenerationModelCapabilityService.ImageModelCapability capability = service.resolveImageCapability(model, "openai_compatible");

        assertFalse(capability.supportsReferenceImages());
        assertEquals(0, capability.minReferenceImages());
        assertEquals(0, capability.maxReferenceImages());
    }

    @Test
    void shouldRejectReferenceImagesForOpenAiImageModel() {
        AiModel model = AiModel.builder()
                .name("GPT Image 1")
                .code("gpt-image-1")
                .build();
        ImageTask task = ImageTask.builder()
                .refImageUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref.png")))
                .build();

        BusinessException ex = assertThrows(BusinessException.class,
                () -> service.validateImageTask(model, task, "openai_compatible"));

        assertTrue(ex.getMessage().contains("不支持参考图输入"));
    }

    @Test
    void shouldUsePresetCapabilityWhenModelConfigDoesNotContainNewFields() {
        GenerationModelCapabilityService serviceWithPreset = new GenerationModelCapabilityService(null, new ModelPresetService() {
            @Override
            public String getPresetConfig(String code) {
                if (!"doubao-seedream-3-0-t2i-250415".equals(code)) {
                    return null;
                }
                return """
                        {
                          "supportReferenceImages": false,
                          "minReferenceImages": 0,
                          "maxReferenceImages": 0
                        }
                        """;
            }
        });

        AiModel model = AiModel.builder()
                .name("Seedream 3.0")
                .code("doubao-seedream-3-0-t2i-250415")
                .config("{\"defaultWidth\":2048,\"defaultHeight\":2048}")
                .build();

        GenerationModelCapabilityService.ImageModelCapability capability = serviceWithPreset.resolveImageCapability(model, "volcengine");

        assertFalse(capability.supportsReferenceImages());
        assertEquals(0, capability.maxReferenceImages());
    }

    @Test
    void shouldRejectFirstFrameForT2vGoogleFlowModel() {
        AiModel model = AiModel.builder()
                .name("Veo T2V Fast")
                .code("veo_3_1_t2v_fast")
                .build();
        VideoTask task = VideoTask.builder()
                .firstFrameImageUrl("https://example.com/first.png")
                .build();

        BusinessException ex = assertThrows(BusinessException.class,
                () -> service.validateVideoTask(model, task, "GoogleFlowReverseApi"));

        assertTrue(ex.getMessage().contains("不支持首帧图输入"));
    }

    @Test
    void shouldRequireTwoImagesForInterpolationModel() {
        AiModel model = AiModel.builder()
                .name("Interpolation Lite")
                .code("veo_3_1_interpolation_lite")
                .build();
        VideoTask task = VideoTask.builder()
                .firstFrameImageUrl("https://example.com/first.png")
                .build();

        BusinessException ex = assertThrows(BusinessException.class,
                () -> service.validateVideoTask(model, task, "GoogleFlowReverseApi"));

        assertTrue(ex.getMessage().contains("至少需要 2 张图片输入"));
    }

    @Test
    void shouldRejectLastFrameForSeedanceProFast() {
        AiModel model = AiModel.builder()
                .name("Seedance 1.0 Pro Fast")
                .code("doubao-seedance-1-0-pro-fast-251015")
                .build();
        VideoTask task = VideoTask.builder()
                .lastFrameImageUrl("https://example.com/last.png")
                .build();

        BusinessException ex = assertThrows(BusinessException.class,
                () -> service.validateVideoTask(model, task, "volcengine"));

        assertTrue(ex.getMessage().contains("不支持尾帧图输入"));
    }

    @Test
    void shouldAllowSeedance20ReferenceMedia() {
        AiModel model = AiModel.builder()
                .name("Seedance 2.0")
                .code("doubao-seedance-2-0-260128")
                .build();
        VideoTask task = VideoTask.builder()
                .firstFrameImageUrl("https://example.com/first.png")
                .lastFrameImageUrl("https://example.com/last.png")
                .referenceImageUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref-1.png")))
                .referenceVideoUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref-1.mp4")))
                .referenceAudioUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref-1.mp3")))
                .build();

        service.validateVideoTask(model, task, "volcengine");
    }

    @Test
    void shouldAllowDashScopeWanImageReferenceImages() {
        AiModel model = AiModel.builder()
                .name("Wan 2.7 Image")
                .code("wan2.7-image")
                .build();
        ImageTask task = ImageTask.builder()
                .refImageUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref.png")))
                .build();

        GenerationModelCapabilityService.ImageModelCapability capability = service.resolveImageCapability(model, "dashscope");

        assertTrue(capability.supportsReferenceImages());
        service.validateImageTask(model, task, "dashscope");
    }

    @Test
    void shouldAllowDashScopeQwenImage20ReferenceImages() {
        AiModel model = AiModel.builder()
                .name("Qwen Image 2.0")
                .code("qwen-image-2.0")
                .build();
        ImageTask task = ImageTask.builder()
                .refImageUrls(JSONUtil.toJsonStr(List.of(
                        "https://example.com/ref-1.png",
                        "https://example.com/ref-2.png")))
                .build();

        GenerationModelCapabilityService.ImageModelCapability capability = service.resolveImageCapability(model, "dashscope");

        assertTrue(capability.supportsReferenceImages());
        assertEquals(2, capability.maxReferenceImages());
        service.validateImageTask(model, task, "dashscope");
    }

    @Test
    void shouldRejectFirstFrameForDashScopeT2vModel() {
        AiModel model = AiModel.builder()
                .name("Wan 2.7 T2V")
                .code("wan2.7-t2v")
                .build();
        VideoTask task = VideoTask.builder()
                .firstFrameImageUrl("https://example.com/first.png")
                .build();

        BusinessException ex = assertThrows(BusinessException.class,
                () -> service.validateVideoTask(model, task, "dashscope"));

        assertTrue(ex.getMessage().contains("不支持首帧图输入"));
    }

    @Test
    void shouldAllowReferenceAudioForDashScopeT2vModel() {
        AiModel model = AiModel.builder()
                .name("Wan 2.7 T2V")
                .code("wan2.7-t2v")
                .build();
        VideoTask task = VideoTask.builder()
                .referenceAudioUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref.mp3")))
                .build();

        GenerationModelCapabilityService.VideoModelCapability capability = service.resolveVideoCapability(model, "dashscope");

        assertTrue(capability.supportsReferenceAudios());
        service.validateVideoTask(model, task, "dashscope");
    }

    @Test
    void shouldAllowFirstAndLastFrameForDashScopeI2vModel() {
        AiModel model = AiModel.builder()
                .name("Wan 2.7 I2V")
                .code("wan2.7-i2v")
                .build();
        VideoTask task = VideoTask.builder()
                .firstFrameImageUrl("https://example.com/first.png")
                .lastFrameImageUrl("https://example.com/last.png")
                .build();

        GenerationModelCapabilityService.VideoModelCapability capability = service.resolveVideoCapability(model, "dashscope");

        assertTrue(capability.supportsFirstFrame());
        assertTrue(capability.supportsLastFrame());
        service.validateVideoTask(model, task, "dashscope");
    }

    @Test
    void shouldAllowReferenceMediaForDashScopeR2vModel() {
        AiModel model = AiModel.builder()
                .name("Wan 2.7 R2V")
                .code("wan2.7-r2v")
                .build();
        VideoTask task = VideoTask.builder()
                .referenceImageUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref.png")))
                .referenceVideoUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref.mp4")))
                .referenceAudioUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref.mp3")))
                .build();

        service.validateVideoTask(model, task, "dashscope");
    }

    @Test
    void shouldAllowDashScopeVideoEditReferenceImageAndVideo() {
        AiModel model = AiModel.builder()
                .name("Wan 2.7 Video Edit")
                .code("wan2.7-videoedit")
                .build();
        VideoTask task = VideoTask.builder()
                .referenceImageUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref.png")))
                .referenceVideoUrls(JSONUtil.toJsonStr(List.of("https://example.com/ref.mp4")))
                .build();

        GenerationModelCapabilityService.VideoModelCapability capability = service.resolveVideoCapability(model, "dashscope");

        assertTrue(capability.supportsReferenceImages());
        assertTrue(capability.supportsReferenceVideos());
        assertFalse(capability.supportsReferenceAudios());
        service.validateVideoTask(model, task, "dashscope");
    }
}