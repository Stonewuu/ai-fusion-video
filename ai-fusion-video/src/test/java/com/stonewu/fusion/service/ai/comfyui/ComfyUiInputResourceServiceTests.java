package com.stonewu.fusion.service.ai.comfyui;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiNativeClient;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiUploadResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ComfyUiInputResourceServiceTests {

    @Mock
    private ComfyUiNativeClient nativeClient;

    @Test
    void uploadImagesDecodesSupportedDataUriBeforeNativeUpload() {
        ComfyUiInputResourceService service = new ComfyUiInputResourceService(nativeClient);
        ApiConfig apiConfig = ApiConfig.builder().id(7L).build();
        byte[] image = "png-data".getBytes(StandardCharsets.UTF_8);
        String source = "data:image/png;base64," + Base64.getEncoder().encodeToString(image);
        when(nativeClient.uploadImage(any(), any(), any(), any(), any()))
                .thenReturn(new ComfyUiUploadResult("task-referenceImages-0.png",
                        "ai-fusion-video", "input"));

        assertThat(service.uploadImages(apiConfig, "task", "referenceImages", List.of(source)))
                .containsExactly("ai-fusion-video/task-referenceImages-0.png");

        ArgumentCaptor<byte[]> bytes = ArgumentCaptor.forClass(byte[].class);
        verify(nativeClient).uploadImage(eq(apiConfig), bytes.capture(),
                eq("task-referenceImages-0.png"), eq("image/png"), eq("ai-fusion-video"));
        assertThat(bytes.getValue()).isEqualTo(image);
    }

    @Test
    void readBoundedStopsBeforeWritingPastConfiguredLimit() {
        byte[] body = "123456789".getBytes(StandardCharsets.UTF_8);

        assertThatThrownBy(() -> ComfyUiInputResourceService.readBounded(
                new ByteArrayInputStream(body), 8))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("不能超过 20MB");
    }
}
