package com.stonewu.fusion.service.generation;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import com.stonewu.fusion.service.system.SystemConfigService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ReferenceImageTransportServiceTests {

    @TempDir
    Path tempDir;

    private final StorageConfigService storageConfigService = mock(StorageConfigService.class);
    private final SystemConfigService systemConfigService = mock(SystemConfigService.class);
    private final PresetArtStyleResourceResolver presetArtStyleResourceResolver =
            mock(PresetArtStyleResourceResolver.class);
    private final AiModel model = AiModel.builder()
            .name("Reference Model")
            .code("reference-model")
            .build();

    private ReferenceImageTransportService service;

    @BeforeEach
    void setUp() {
        when(presetArtStyleResourceResolver.isPresetArtStylePath(anyString())).thenReturn(false);
        service = new ReferenceImageTransportService(
                storageConfigService,
                systemConfigService,
                presetArtStyleResourceResolver);
    }

    @Test
    void prefersPublicUrlWhenUrlAndDataUriAreBothAllowed() {
        when(systemConfigService.resolvePublicUrl("/media/reference.png"))
                .thenReturn("https://fusion.example.com/media/reference.png");

        List<String> resolved = service.resolveInputs(
                model,
                config("url", "data_uri"),
                List.of("/media/reference.png"),
                null);

        assertThat(resolved).containsExactly("https://fusion.example.com/media/reference.png");
    }

    @Test
    void convertsLocalImageToDataUriWhenPublicUrlIsUnavailable() throws IOException {
        byte[] imageBytes = new byte[]{1, 2, 3, 4};
        Files.write(tempDir.resolve("reference.png"), imageBytes);
        when(storageConfigService.getDefaultConfig()).thenReturn(StorageConfig.builder()
                .basePath(tempDir.toString())
                .build());

        List<String> resolved = service.resolveInputs(
                model,
                config("url", "data_uri"),
                List.of("/media/reference.png"),
                null);

        assertThat(resolved).containsExactly(
                "data:image/png;base64," + Base64.getEncoder().encodeToString(imageBytes));
    }

    @Test
    void rejectsReferenceImagesWhenNoTransportModeIsConfigured() {
        assertThatThrownBy(() -> service.validateInputs(
                model,
                JSONUtil.createObj(),
                List.of("https://example.com/reference.png")))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("未配置允许的参考图传递模式");
    }

    @Test
    void rejectsLocalImageForUrlOnlyModelWithoutPublicAccess() {
        assertThatThrownBy(() -> service.validateInputs(
                model,
                config("url"),
                List.of("/media/reference.png")))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("未配置访问域名或公网对象存储");
    }

    @Test
    void rejectsLocalMediaPathTraversalDuringDataUriConversion() throws IOException {
        Path mediaRoot = Files.createDirectory(tempDir.resolve("media"));
        Files.write(tempDir.resolve("secret.png"), new byte[]{9, 8, 7});
        when(storageConfigService.getDefaultConfig()).thenReturn(StorageConfig.builder()
                .basePath(mediaRoot.toString())
                .build());

        assertThatThrownBy(() -> service.resolveInputs(
                model,
                config("data_uri"),
                List.of("/media/../secret.png"),
                null))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("本地参考图不存在");
    }

    private JSONObject config(String... formats) {
        return JSONUtil.createObj().set("referenceImageInputFormats", List.of(formats));
    }
}
