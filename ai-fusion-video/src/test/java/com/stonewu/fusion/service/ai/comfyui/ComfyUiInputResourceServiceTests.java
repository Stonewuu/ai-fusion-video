package com.stonewu.fusion.service.ai.comfyui;

import com.stonewu.fusion.common.BusinessException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.net.InetSocketAddress;
import java.util.concurrent.atomic.AtomicReference;
import java.util.Base64;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verifyNoInteractions;

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

    @Test
    void audioDataUriUsesTemporaryFileAndRemovesItAfterUpload() throws Exception {
        var service = new ComfyUiInputResourceService(nativeClient);
        byte[] audio = "audio-data".getBytes(StandardCharsets.UTF_8);
        String source = "data:audio/wav;base64," + Base64.getEncoder().encodeToString(audio);
        AtomicReference<Path> temporary = new AtomicReference<>();
        when(nativeClient.uploadFile(any(), any(), any(), any(), any())).thenAnswer(invocation -> {
            Path path = invocation.getArgument(1);
            temporary.set(path);
            assertThat(Files.readAllBytes(path)).isEqualTo(audio);
            assertThat(invocation.<String>getArgument(2)).isEqualTo("task-referenceAudios-0.wav");
            assertThat(invocation.<String>getArgument(3)).isEqualTo("audio/wav");
            return new ComfyUiUploadResult("voice.wav", "input-files", "input");
        });
        assertThat(service.uploadMedia(ApiConfig.builder().build(), "task", "referenceAudios", List.of(source), "audio"))
                .containsExactly("input-files/voice.wav");
        assertThat(Files.exists(temporary.get())).isFalse();
    }

    @Test
    void failedMediaUploadStillRemovesItsTemporaryFile() throws Exception {
        var service = new ComfyUiInputResourceService(nativeClient);
        String source = "data:video/mp4;base64," + Base64.getEncoder().encodeToString(new byte[]{1, 2, 3});
        AtomicReference<Path> temporary = new AtomicReference<>();
        when(nativeClient.uploadFile(any(), any(), any(), any(), any())).thenAnswer(invocation -> {
            temporary.set(invocation.getArgument(1));
            throw new BusinessException(502, "上传失败");
        });
        assertThatThrownBy(() -> service.uploadMedia(ApiConfig.builder().build(), "task", "referenceVideos",
                List.of(source), "video")).isInstanceOf(BusinessException.class).hasMessage("上传失败");
        assertThat(Files.exists(temporary.get())).isFalse();
    }

    @Test
    void rejectsWrongMediaKindAndMalformedBase64BeforeNativeUpload() {
        var service = new ComfyUiInputResourceService(nativeClient);
        var api = ApiConfig.builder().build();
        assertThatThrownBy(() -> service.uploadMedia(api, "task", "referenceVideos",
                List.of("data:audio/wav;base64,AQID"), "video"))
                .isInstanceOf(BusinessException.class).hasMessageContaining("类型与绑定不匹配");
        assertThatThrownBy(() -> service.uploadMedia(api, "task", "referenceAudios",
                List.of("data:audio/wav;base64,%%%"), "audio"))
                .isInstanceOf(BusinessException.class).hasMessageContaining("base64 无效");
        assertThatThrownBy(() -> service.uploadMedia(api, "task", "referenceAudios",
                List.of("data:audio/wav;base64,AQ==junk"), "audio"))
                .isInstanceOf(BusinessException.class).hasMessageContaining("base64 无效");
        verifyNoInteractions(nativeClient);
    }

    @Test
    void videoHttpInputIsDownloadedThenUploadedWithItsMimeAndServerDescriptor() throws Exception {
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.start();
        AtomicReference<String> sourceAuthorization = new AtomicReference<>();
        AtomicReference<String> uploadAuthorization = new AtomicReference<>();
        AtomicReference<String> multipart = new AtomicReference<>();
        try {
            server.createContext("/reference.mp4", exchange -> {
                sourceAuthorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
                exchange.getResponseHeaders().set("Content-Type", "video/mp4");
                exchange.sendResponseHeaders(200, 0);
                try (var output = exchange.getResponseBody()) {
                    output.write("video-input-bytes".getBytes(StandardCharsets.UTF_8));
                }
            });
            server.createContext("/upload/image", exchange -> {
                uploadAuthorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
                multipart.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
                byte[] response = "{\"name\":\"uploaded.mp4\",\"subfolder\":\"inputs\",\"type\":\"input\"}"
                        .getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, response.length);
                try (var output = exchange.getResponseBody()) {
                    output.write(response);
                }
            });
            String base = "http://127.0.0.1:" + server.getAddress().getPort();
            var api = ApiConfig.builder().apiUrl(base).apiKey("test-token").build();
            var service = new ComfyUiInputResourceService(new ComfyUiNativeClient(new ObjectMapper()));
            assertThat(service.uploadMedia(api, "task", "referenceVideos", List.of(base + "/reference.mp4"), "video"))
                    .containsExactly("inputs/uploaded.mp4");
            assertThat(sourceAuthorization.get()).isNull();
            assertThat(uploadAuthorization.get()).isEqualTo("Bearer test-token");
            assertThat(multipart.get()).contains("name=\"image\"; filename=\"task-referenceVideos-0.mp4\"",
                    "Content-Type: video/mp4", "video-input-bytes");
        } finally {
            server.stop(0);
        }
    }
}
