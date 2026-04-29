package com.stonewu.fusion.service.generation.strategy.impl;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.generation.ImageGenerationService;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OpenAiImageStrategyTests {

    private HttpServer server;
        private final StorageConfigService storageConfigService = mock(StorageConfigService.class);

        @Test
        void getNameUsesOpenAiCompatiblePlatformKey() {
                OpenAiImageStrategy strategy = new OpenAiImageStrategy(
                                mock(ImageGenerationService.class),
                                mock(AiModelService.class),
                                mock(ModelPresetService.class),
                                mock(MediaStorageService.class),
                                storageConfigService
                );

                assertThat(strategy.getName()).isEqualTo("openai_compatible");
        }

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void generateSupportsOpenAiCompatibleImagePathAndUrlResponse() throws Exception {
        AtomicReference<String> requestPath = new AtomicReference<>();
        AtomicReference<String> requestBody = new AtomicReference<>();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/images/generations", exchange -> {
            requestPath.set(exchange.getRequestURI().getPath());
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            writeJson(exchange, 200, """
                    {"data":[{"url":"https://example.com/generated.png"}]}
                    """);
        });
        server.start();

        OpenAiImageStrategy strategy = new OpenAiImageStrategy(
                mock(ImageGenerationService.class),
                mock(AiModelService.class),
                mock(ModelPresetService.class),
                mock(MediaStorageService.class),
                storageConfigService
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai_compatible")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .autoAppendV1Path(false)
                .build();

        List<String> urls = strategy.generate("a test prompt", "gpt-image-2", 1024, 1024, 1, null, apiConfig);

        assertThat(urls).containsExactly("https://example.com/generated.png");
        assertThat(requestPath.get()).isEqualTo("/images/generations");
        assertThat(requestBody.get()).contains("\"model\":\"gpt-image-2\"");
        assertThat(requestBody.get()).contains("\"size\":\"1024x1024\"");
    }

    @Test
    void generateStoresBase64ImageResponses() throws Exception {
        byte[] imageBytes = "fake-image".getBytes(StandardCharsets.UTF_8);
        String base64Image = Base64.getEncoder().encodeToString(imageBytes);

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/images/generations", exchange -> writeJson(exchange, 200, """
                {"data":[{"b64_json":"%s"}]}
                """.formatted(base64Image)));
        server.start();

        MediaStorageService mediaStorageService = mock(MediaStorageService.class);
        when(mediaStorageService.storeBytes(any(), eq("images"), eq("png")))
                .thenReturn("/media/images/generated.png");

        OpenAiImageStrategy strategy = new OpenAiImageStrategy(
                mock(ImageGenerationService.class),
                mock(AiModelService.class),
                mock(ModelPresetService.class),
                mediaStorageService,
                storageConfigService
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .build();

        List<String> urls = strategy.generate("a test prompt", "gpt-image-2", 1024, 1024, 1, null, apiConfig);

        assertThat(urls).containsExactly("/media/images/generated.png");
        verify(mediaStorageService).storeBytes(imageBytes, "images", "png");
    }

    @Test
    void generateAllowsConfiguredCustomImageResolution() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/images/generations", exchange -> {
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            writeJson(exchange, 200, """
                    {"data":[{"url":"https://example.com/generated-2k.png"}]}
                    """);
        });
        server.start();

        ModelPresetService modelPresetService = mock(ModelPresetService.class);
        when(modelPresetService.getPresetConfig("gpt-image-2")).thenReturn("""
                {
                  "supportCustomSize": true,
                  "sizeMultiple": 16,
                  "maxEdge": 3840,
                  "maxAspectRatio": 3,
                  "minPixels": 655360,
                  "maxPixels": 8294400,
                  "defaultWidth": 1024,
                  "defaultHeight": 1024
                }
                """);

        OpenAiImageStrategy strategy = new OpenAiImageStrategy(
                mock(ImageGenerationService.class),
                mock(AiModelService.class),
                modelPresetService,
                mock(MediaStorageService.class),
                storageConfigService
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai_compatible")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .autoAppendV1Path(false)
                .build();

        List<String> urls = strategy.generate("a scenic panorama", "gpt-image-2", 2560, 1440, 1, null, apiConfig);

        assertThat(urls).containsExactly("https://example.com/generated-2k.png");
        assertThat(requestBody.get()).contains("\"size\":\"2560x1440\"");
    }

    @Test
    void submitUsesImagesGenerateApiForGptImage2() throws Exception {
        byte[] imageBytes = "generated-image".getBytes(StandardCharsets.UTF_8);
        String base64Image = Base64.getEncoder().encodeToString(imageBytes);
        AtomicReference<String> requestPath = new AtomicReference<>();
        AtomicReference<String> requestBody = new AtomicReference<>();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/images/generations", exchange -> {
            requestPath.set(exchange.getRequestURI().getPath());
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            writeJson(exchange, 200, """
                    {"data":[{"b64_json":"%s"}],"output_format":"png"}
                    """.formatted(base64Image));
        });
        server.start();

        MediaStorageService mediaStorageService = mock(MediaStorageService.class);
        when(mediaStorageService.storeBytes(any(), eq("images"), eq("png")))
                .thenReturn("/media/images/generated.png");
        ImageGenerationService imageGenerationService = mock(ImageGenerationService.class);
        AiModelService aiModelService = mock(AiModelService.class);
        when(aiModelService.getById(100L)).thenReturn(AiModel.builder()
                .id(100L)
                .code("gpt-image-2")
                .config("{\"defaultWidth\":1024,\"defaultHeight\":1024}")
                .build());
        when(imageGenerationService.listItems(200L)).thenReturn(List.of(ImageItem.builder().id(300L).status(0).build()));

        OpenAiImageStrategy strategy = new OpenAiImageStrategy(
                imageGenerationService,
                aiModelService,
                mock(ModelPresetService.class),
                mediaStorageService,
                storageConfigService
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .build();
        ImageTask task = ImageTask.builder()
                .id(200L)
                .taskId("task-1")
                .modelId(100L)
                .prompt("a test prompt")
                .build();

        String platformTaskId = strategy.submit(task, apiConfig);

        assertThat(platformTaskId).isEqualTo("task-1");
        assertThat(requestPath.get()).isEqualTo("/v1/images/generations");
        assertThat(requestBody.get()).contains("\"model\":\"gpt-image-2\"");
        assertThat(requestBody.get()).contains("\"size\":\"1024x1024\"");
        verify(mediaStorageService).storeBytes(imageBytes, "images", "png");
        verify(imageGenerationService).updateItem(any(ImageItem.class));
        verify(imageGenerationService).update(task);
    }

    @Test
    void generateUsesImagesEditEndpointWhenReferenceImagesProvided() throws Exception {
        byte[] referenceBytes = "reference-image".getBytes(StandardCharsets.UTF_8);
        byte[] generatedBytes = "edited-image".getBytes(StandardCharsets.UTF_8);
        String base64Image = Base64.getEncoder().encodeToString(generatedBytes);
        AtomicReference<String> requestPath = new AtomicReference<>();
        AtomicReference<String> requestContentType = new AtomicReference<>();
        AtomicReference<String> requestBody = new AtomicReference<>();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/reference.png", exchange -> {
            exchange.getResponseHeaders().add("Content-Type", "image/png");
            exchange.sendResponseHeaders(200, referenceBytes.length);
            exchange.getResponseBody().write(referenceBytes);
            exchange.close();
        });
        server.createContext("/v1/images/edits", exchange -> {
            requestPath.set(exchange.getRequestURI().getPath());
            requestContentType.set(exchange.getRequestHeaders().getFirst("Content-Type"));
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            writeJson(exchange, 200, """
                    {"data":[{"b64_json":"%s"}],"output_format":"png"}
                    """.formatted(base64Image));
        });
        server.start();

        MediaStorageService mediaStorageService = mock(MediaStorageService.class);
        when(mediaStorageService.storeBytes(any(), eq("images"), eq("png")))
                .thenReturn("/media/images/edited.png");

        OpenAiImageStrategy strategy = new OpenAiImageStrategy(
                mock(ImageGenerationService.class),
                mock(AiModelService.class),
                mock(ModelPresetService.class),
                mediaStorageService,
                storageConfigService
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .build();

        List<String> urls = strategy.generate(
                "edit this image",
                "gpt-image-1",
                1024,
                1024,
                1,
                List.of("http://localhost:" + server.getAddress().getPort() + "/reference.png"),
                apiConfig
        );

        assertThat(urls).containsExactly("/media/images/edited.png");
        assertThat(requestPath.get()).isEqualTo("/v1/images/edits");
        assertThat(requestContentType.get()).contains("multipart/form-data");
        assertThat(requestBody.get()).contains("name=\"image[]\"");
        assertThat(requestBody.get()).contains("name=\"model\"");
        assertThat(requestBody.get()).contains("gpt-image-1");
        assertThat(requestBody.get()).contains("name=\"prompt\"");
        assertThat(requestBody.get()).contains("edit this image");
        verify(mediaStorageService).storeBytes(generatedBytes, "images", "png");
    }

    @Test
    void generateUsesConfiguredAsyncTaskMode() throws Exception {
        AtomicReference<String> generationPath = new AtomicReference<>();
        AtomicReference<String> taskPath = new AtomicReference<>();
        AtomicReference<String> generationBody = new AtomicReference<>();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/images/generations", exchange -> {
            generationPath.set(exchange.getRequestURI().getPath());
            generationBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            writeJson(exchange, 200, """
                    {"code":200,"data":[{"status":"submitted","task_id":"task-123"}]}
                    """);
        });
        server.createContext("/v1/tasks/task-123", exchange -> {
            taskPath.set(exchange.getRequestURI().getPath());
            writeJson(exchange, 200, """
                    {"code":200,"data":{"id":"task-123","status":"completed","progress":100,"result":{"images":[{"url":["https://example.com/async.png"]}]}}}
                    """);
        });
        server.start();

        ModelPresetService modelPresetService = mock(ModelPresetService.class);
        when(modelPresetService.getPresetConfig("gpt-image-2-official")).thenReturn("""
                {
                  "asyncMode": true,
                  "asyncTaskInitialDelaySeconds": 0,
                  "asyncTaskPollIntervalSeconds": 0,
                  "asyncTaskTimeoutSeconds": 5,
                  "defaultWidth": 1024,
                  "defaultHeight": 1024
                }
                """);

        OpenAiImageStrategy strategy = new OpenAiImageStrategy(
                mock(ImageGenerationService.class),
                mock(AiModelService.class),
                modelPresetService,
                mock(MediaStorageService.class),
                storageConfigService
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai_compatible")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .build();

        List<String> urls = strategy.generate(
                "a test prompt",
                "gpt-image-2-official",
                1024,
                1024,
                1,
                List.of("https://example.com/reference.png"),
                apiConfig
        );

        assertThat(urls).containsExactly("https://example.com/async.png");
        assertThat(generationPath.get()).isEqualTo("/v1/images/generations");
        assertThat(taskPath.get()).isEqualTo("/v1/tasks/task-123");
        assertThat(generationBody.get()).contains("\"image_urls\":[\"https://example.com/reference.png\"]");
        assertThat(generationBody.get()).doesNotContain("asyncMode");
    }

    private static void writeJson(HttpExchange exchange, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }
}