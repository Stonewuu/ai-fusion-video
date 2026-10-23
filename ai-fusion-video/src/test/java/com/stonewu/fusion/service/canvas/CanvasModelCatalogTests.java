package com.stonewu.fusion.service.canvas;

import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CanvasModelCatalogTests {
    private final AiModelService models = mock(AiModelService.class);
    private final ApiConfigService apis = mock(ApiConfigService.class);
    private final GenerationModelCapabilityService capabilities = mock(GenerationModelCapabilityService.class);
    private final CanvasModelCatalog catalog = new CanvasModelCatalog(models, apis, capabilities,
            mock(ComfyUiWorkflowService.class), new ObjectMapper());

    @Test void exposesOnlyDeclaredInputConstraintsWithoutCredentialsOrRawConfig() throws Exception {
        AiModel model = model();
        ApiConfig api = new ApiConfig(); api.setId(10L); api.setStatus(1); api.setApiKey("test-only-sensitive-value");
        when(apis.getById(10L)).thenReturn(api);
        when(models.getEnabledList()).thenReturn(List.of(model));
        when(capabilities.buildImageCapabilitySnapshot(model)).thenReturn(JSONUtil.createObj()
                .set("supportedAspectRatios", List.of("1:1", "16:9")).set("apiKey", "must-not-project"));
        when(capabilities.resolveImageCapability(model)).thenReturn(new GenerationModelCapabilityService.ImageModelCapability(
                true, 0, 4, List.of("url"), true, false));
        var result = catalog.list();
        assertThat(result.getFirst().inputs().get("referenceImages").maximum()).isEqualTo(4);
        assertThat(result.getFirst().parameters().get("supportedAspectRatios")).isEqualTo(List.of("1:1", "16:9"));
        String serialized = new ObjectMapper().writeValueAsString(result);
        assertThat(serialized).doesNotContain("test-only-sensitive-value", "must-not-project", "raw-config-secret", "apiKey");
    }

    @Test void readsSupplierAvailabilityAgainOnEveryValidation() {
        AiModel model = model();
        when(models.getById(1L)).thenReturn(model);
        ApiConfig api = new ApiConfig(); api.setId(10L); api.setStatus(0);
        when(apis.getById(10L)).thenReturn(api);
        assertThatThrownBy(() -> catalog.require(1)).hasMessageContaining("停用");
        api.setStatus(1);
        when(capabilities.buildImageCapabilitySnapshot(model)).thenReturn(JSONUtil.createObj());
        when(capabilities.resolveImageCapability(model)).thenReturn(new GenerationModelCapabilityService.ImageModelCapability(
                false, 0, 0, List.of(), false, false));
        assertThat(catalog.require(1).inputs()).doesNotContainKey("referenceImages");
        model.setStatus(0);
        assertThatThrownBy(() -> catalog.require(1)).hasMessageContaining("停用");
    }

    @Test void audioPickerExposesOnlyNamesAndRejectsDisabledOrWrongProviders() throws Exception {
        ApiConfig api = new ApiConfig(); api.setId(12L); api.setName("配乐服务");
        api.setPlatform("sonilo"); api.setStatus(1); api.setApiKey("test-only-audio-key");
        when(apis.getListByPlatform("sonilo")).thenReturn(List.of(api));
        when(apis.getById(12L)).thenReturn(api);
        assertThat(catalog.audioServices()).containsExactly(new CanvasModelCatalog.AudioService(12, "配乐服务"));
        assertThat(new ObjectMapper().writeValueAsString(catalog.audioServices())).doesNotContain("test-only-audio-key");
        catalog.requireAudio(12);
        api.setStatus(0);
        assertThatThrownBy(() -> catalog.requireAudio(12)).hasMessageContaining("Sonilo");
        api.setStatus(1); api.setPlatform("openai");
        assertThatThrownBy(() -> catalog.requireAudio(12)).hasMessageContaining("Sonilo");
    }

    private AiModel model() {
        AiModel model = new AiModel(); model.setId(1L); model.setName("测试图片模型"); model.setModelType(2);
        model.setStatus(1); model.setApiConfigId(10L); model.setModelProtocol("openai");
        model.setConfig("{\"apiKey\":\"raw-config-secret\"}");
        return model;
    }
}
