package com.stonewu.fusion.service.generation;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import com.stonewu.fusion.service.system.SystemConfigService;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.InetSocketAddress;
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
    private HttpServer server;

    @BeforeEach
    void setUp() {
        when(presetArtStyleResourceResolver.isPresetArtStylePath(anyString())).thenReturn(false);
        service = new ReferenceImageTransportService(
                storageConfigService,
                systemConfigService,
                presetArtStyleResourceResolver);
    }

    @AfterEach
    void tearDown() {
        if (server != null) server.stop(0);
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
    void convertsLoopbackHttpUrlToDataUriEvenWhenUrlIsAllowed() throws IOException {
        byte[] imageBytes = new byte[]{5, 6, 7, 8};
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/reference.png", exchange -> {
            exchange.getResponseHeaders().set("Content-Type", "image/png");
            exchange.sendResponseHeaders(200, imageBytes.length);
            exchange.getResponseBody().write(imageBytes);
            exchange.close();
        });
        server.start();

        List<String> resolved = service.resolveInputs(
                model,
                config("url", "data_uri"),
                List.of("http://127.0.0.1:" + server.getAddress().getPort() + "/reference.png"),
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
    void rejectsLoopbackAndPrivateHttpUrlsForUrlOnlyModel() {
        assertThatThrownBy(() -> service.validateInputs(
                model,
                config("url"),
                List.of("http://localhost:3000/reference.png")))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("远端模型无法访问");

        assertThatThrownBy(() -> service.validateInputs(
                model,
                config("url"),
                List.of("http://192.168.1.20/reference.png")))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("远端模型无法访问");
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
