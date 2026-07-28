package com.stonewu.fusion.controller.storage;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.storage.StorageTypes;
import com.stonewu.fusion.service.system.SystemConfigService;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class FileUploadControllerTests {

    private final MediaStorageService mediaStorageService = mock(MediaStorageService.class);
    private final StorageConfigService storageConfigService = mock(StorageConfigService.class);
    private final SystemConfigService systemConfigService = mock(SystemConfigService.class);
    private final AiModelService aiModelService = mock(AiModelService.class);
    private final FileUploadController controller = new FileUploadController(
            mediaStorageService, storageConfigService, systemConfigService, aiModelService);

    @Test
    void uploadsSupportedUrlInputAndReturnsPublicUrl() throws Exception {
        AiModel model = enabledModel(List.of("image"), Map.of("image", List.of("url")));
        MockMultipartFile file = new MockMultipartFile(
                "file", "sample.png", "image/png", "image".getBytes());
        when(aiModelService.getById(7L)).thenReturn(model);
        when(storageConfigService.getDefaultConfig()).thenReturn(
                StorageConfig.builder().type(StorageTypes.S3).build());
        when(mediaStorageService.storeBytes(file.getBytes(), "assistant/image", "png"))
                .thenReturn("stored/sample.png");
        when(systemConfigService.resolvePublicUrl("stored/sample.png"))
                .thenReturn("https://cdn.example.com/sample.png");

        var result = controller.uploadAssistantInput(file, 7L, "url");

        assertThat(result.getData()).isEqualTo("https://cdn.example.com/sample.png");
        verify(mediaStorageService).storeBytes(file.getBytes(), "assistant/image", "png");
    }

    @Test
    void rejectsUrlUploadWhenModelOnlySupportsBase64() {
        AiModel model = enabledModel(List.of("image"), Map.of("image", List.of("base64")));
        MockMultipartFile file = new MockMultipartFile(
                "file", "sample.png", "image/png", "image".getBytes());
        when(aiModelService.getById(8L)).thenReturn(model);

        assertThatThrownBy(() -> controller.uploadAssistantInput(file, 8L, "url"))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("不支持 image 的 url 输入");
        verifyNoInteractions(mediaStorageService);
    }

    @Test
    void persistsBase64InputForMessagePreviewWithoutPublicSiteUrl() throws Exception {
        AiModel model = enabledModel(List.of("image"), Map.of("image", List.of("base64")));
        MockMultipartFile file = new MockMultipartFile(
                "file", "sample.png", "image/png", "image".getBytes());
        when(aiModelService.getById(9L)).thenReturn(model);
        when(storageConfigService.getDefaultConfig()).thenReturn(
                StorageConfig.builder().type(StorageTypes.LOCAL).build());
        when(mediaStorageService.storeBytes(file.getBytes(), "assistant/image", "png"))
                .thenReturn("/media/assistant/image/sample.png");

        var result = controller.uploadAssistantInput(file, 9L, "base64");

        assertThat(result.getData()).isEqualTo("/media/assistant/image/sample.png");
        verify(mediaStorageService).storeBytes(file.getBytes(), "assistant/image", "png");
        verifyNoInteractions(systemConfigService);
    }

    @Test
    void rejectsLocalUrlUploadWithoutExplicitPublicResourceUrl() {
        AiModel model = enabledModel(List.of("image"), Map.of("image", List.of("url")));
        MockMultipartFile file = new MockMultipartFile(
                "file", "sample.png", "image/png", "image".getBytes());
        when(aiModelService.getById(10L)).thenReturn(model);
        when(storageConfigService.getDefaultConfig()).thenReturn(
                StorageConfig.builder().type(StorageTypes.LOCAL).build());
        when(systemConfigService.getPublicResourceBaseUrl()).thenReturn(null);

        assertThatThrownBy(() -> controller.uploadAssistantInput(file, 10L, "url"))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("后端资源公网地址");

        verifyNoInteractions(mediaStorageService);
    }

    private AiModel enabledModel(List<String> types, Map<String, List<String>> transports) {
        return AiModel.builder()
                .modelType(1)
                .status(1)
                .multimodalInputTypes(types)
                .multimodalInputTransports(transports)
                .build();
    }
}
