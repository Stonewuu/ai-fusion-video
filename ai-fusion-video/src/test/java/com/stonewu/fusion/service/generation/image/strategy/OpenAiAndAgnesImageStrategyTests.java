package com.stonewu.fusion.service.generation.image.strategy;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.agnes.AgnesImageStrategy;
import com.stonewu.fusion.service.generation.image.strategy.newapi.NewApiImageStrategy;
import com.stonewu.fusion.service.generation.image.strategy.openai.OpenAiImageStrategy;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OpenAiAndAgnesImageStrategyTests {

    private static final String AGNES_IMAGE_CONFIG = """
            {
              "imageProtocol": "agnes",
              "supportDataUriInput": true,
              "referenceImageInputFormats": ["url", "data_uri"],
              "singleImagePerRequest": true,
              "defaultSizeTier": "1K",
              "defaultAspectRatio": "1:1",
              "defaultWidth": 1024,
              "defaultHeight": 1024,
              "responseFormat": "url",
              "supportedResolutions": ["1K", "2K"],
              "supportedAspectRatios": ["1:1", "16:9"],
              "supportedSizes": {
                "1K": {"1:1": "1024x1024"},
                "2K": {"16:9": "2624x1472"}
              }
            }
            """;

    private static final String AGNES_IMAGE_20_CONFIG = """
            {
              "imageProtocol": "agnes",
              "supportDataUriInput": true,
              "referenceImageInputFormats": ["url", "data_uri"],
              "singleImagePerRequest": true,
              "defaultWidth": 1024,
              "defaultHeight": 1024,
              "responseFormat": "url",
              "supportedResolutions": ["1024x1024", "1024x768", "768x1024"],
              "supportedAspectRatios": ["1:1", "4:3", "3:4"],
              "supportedSizes": {
                "standard": {
                  "1:1": "1024x1024",
                  "4:3": "1024x768",
                  "3:4": "768x1024"
                }
              }
            }
            """;

    private HttpServer server;
                private final StorageConfigService storageConfigService = mock(StorageConfigService.class);
                private final PresetArtStyleResourceResolver presetArtStyleResourceResolver = new PresetArtStyleResourceResolver();

        @Test
    void getNameUsesOfficialOpenAiProtocolKey() {
                OpenAiImageStrategy strategy = new OpenAiImageStrategy(
                                mock(ImageGenerationService.class),
                                mock(AiModelService.class),
                                mock(ModelPresetService.class),
                                mock(MediaStorageService.class),
                                storageConfigService,
                                presetArtStyleResourceResolver
                );

                assertThat(strategy.getName()).isEqualTo("openai");
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
                storageConfigService,
                presetArtStyleResourceResolver
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
                storageConfigService,
                presetArtStyleResourceResolver
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
                storageConfigService,
                presetArtStyleResourceResolver
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai_compatible")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .autoAppendV1Path(false)
                .build();

        AiModel model = AiModel.builder()
                .code("gpt-image-2")
                .capabilityPresetCode("gpt-image-2")
                .build();
        List<String> urls = strategy.generate("a scenic panorama", model, 2560, 1440, 1, null, apiConfig);

        assertThat(urls).containsExactly("https://example.com/generated-2k.png");
        assertThat(requestBody.get()).contains("\"size\":\"2560x1440\"");
    }

    @Test
    void generateUsesAgnesGenerationSchemaAndBatchesCount() throws Exception {
        AtomicReference<String> requestPath = new AtomicReference<>();
        List<String> requestBodies = new ArrayList<>();
        AtomicInteger requestCount = new AtomicInteger();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/images/generations", exchange -> {
            requestPath.set(exchange.getRequestURI().getPath());
            requestBodies.add(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            int currentRequest = requestCount.incrementAndGet();
            writeJson(exchange, 200, """
                    {"data":[{"url":"https://example.com/agnes-%d.png"}]}
                    """.formatted(currentRequest));
        });
        server.start();

        ModelPresetService modelPresetService = mock(ModelPresetService.class);
        when(modelPresetService.getPresetConfig("agnes-image-2.1-flash")).thenReturn(AGNES_IMAGE_CONFIG);

        AgnesImageStrategy strategy = new AgnesImageStrategy(
                mock(ImageGenerationService.class),
                mock(AiModelService.class),
                modelPresetService,
                mock(MediaStorageService.class),
                storageConfigService,
                presetArtStyleResourceResolver
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("newapi")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .build();

        List<String> urls = strategy.generate(
                "a cinematic city",
                AiModel.builder()
                        .code("agnes-image-2.1-flash")
                        .capabilityPresetCode("agnes-image-2.1-flash")
                        .build(),
                2624,
                1472,
                2,
                List.of(
                        "https://example.com/reference.png",
                        "data:image/png;base64,cmVmZXJlbmNl"
                ),
                apiConfig
        );

        assertThat(urls).containsExactly(
                "https://example.com/agnes-1.png",
                "https://example.com/agnes-2.png"
        );
        assertThat(requestPath.get()).isEqualTo("/v1/images/generations");
        assertThat(requestCount.get()).isEqualTo(2);
        assertThat(requestBodies).hasSize(2);
        for (String requestBody : requestBodies) {
            JSONObject json = JSONUtil.parseObj(requestBody);
            assertThat(json.getStr("model")).isEqualTo("agnes-image-2.1-flash");
            assertThat(json.getStr("size")).isEqualTo("2K");
            assertThat(json.getStr("ratio")).isEqualTo("16:9");
            assertThat(json.containsKey("n")).isFalse();
            assertThat(json.containsKey("response_format")).isFalse();
            assertThat(json.getJSONObject("extra_body").getStr("response_format")).isEqualTo("url");
            assertThat(json.getJSONObject("extra_body").getJSONArray("image").toList(String.class))
                    .containsExactly(
                            "https://example.com/reference.png",
                            "data:image/png;base64,cmVmZXJlbmNl"
                    );
        }
    }

    @Test
    void generateUsesSharedAgnesProtocolFor20AndPreservesDataUriInput() throws Exception {
        AtomicReference<String> requestPath = new AtomicReference<>();
        AtomicReference<String> requestBody = new AtomicReference<>();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/images/generations", exchange -> {
            requestPath.set(exchange.getRequestURI().getPath());
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            writeJson(exchange, 200, """
                    {"data":[{"url":"https://example.com/agnes-20.png"}]}
                    """);
        });
        server.start();

        ModelPresetService modelPresetService = mock(ModelPresetService.class);
        when(modelPresetService.getPresetConfig("agnes-image-2.0-flash")).thenReturn(AGNES_IMAGE_20_CONFIG);

        AgnesImageStrategy strategy = new AgnesImageStrategy(
                mock(ImageGenerationService.class),
                mock(AiModelService.class),
                modelPresetService,
                mock(MediaStorageService.class),
                storageConfigService,
                presetArtStyleResourceResolver
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai_compatible")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .build();

        List<String> urls = strategy.generate(
                "combine the references",
                AiModel.builder()
                        .code("agnes-image-2.0-flash")
                        .capabilityPresetCode("agnes-image-2.0-flash")
                        .build(),
                1024,
                768,
                1,
                List.of("data:image/png;base64,cmVmZXJlbmNl"),
                apiConfig
        );

        JSONObject json = JSONUtil.parseObj(requestBody.get());
        assertThat(urls).containsExactly("https://example.com/agnes-20.png");
        assertThat(requestPath.get()).isEqualTo("/v1/images/generations");
        assertThat(json.getStr("size")).isEqualTo("1024x768");
        assertThat(json.containsKey("ratio")).isFalse();
        assertThat(json.containsKey("n")).isFalse();
        assertThat(json.containsKey("response_format")).isFalse();
        assertThat(json.containsKey("tags")).isFalse();
        assertThat(json.getJSONObject("extra_body").getStr("response_format")).isEqualTo("url");
        assertThat(json.getJSONObject("extra_body").getJSONArray("image").toList(String.class))
                .containsExactly("data:image/png;base64,cmVmZXJlbmNl");
    }

    @Test
    void submitMapsAgnesTaskTierAndRatioToNativeDimensions() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/images/generations", exchange -> {
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            writeJson(exchange, 200, """
                    {"data":[{"url":"https://example.com/agnes-2k.png"}]}
                    """);
        });
        server.start();

        ImageGenerationService imageGenerationService = mock(ImageGenerationService.class);
        AiModelService aiModelService = mock(AiModelService.class);
        ModelPresetService modelPresetService = mock(ModelPresetService.class);
        when(modelPresetService.getPresetConfig("agnes-image-2.1-flash")).thenReturn(AGNES_IMAGE_CONFIG);
        when(aiModelService.getById(101L)).thenReturn(AiModel.builder()
                .id(101L)
                .code("agnes-image-2.1-flash")
                .capabilityPresetCode("agnes-image-2.1-flash")
                .build());
        when(imageGenerationService.listItems(201L)).thenReturn(List.of(
                ImageItem.builder().id(301L).status(0).build()
        ));

        AgnesImageStrategy strategy = new AgnesImageStrategy(
                imageGenerationService,
                aiModelService,
                modelPresetService,
                mock(MediaStorageService.class),
                storageConfigService,
                presetArtStyleResourceResolver
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai_compatible")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .build();
        ImageTask task = ImageTask.builder()
                .id(201L)
                .taskId("agnes-task")
                .modelId(101L)
                .prompt("a wide cinematic landscape")
                .resolution("2K")
                .ratio("16:9")
                .count(1)
                .build();

        String platformTaskId = strategy.submit(task, apiConfig);

        JSONObject json = JSONUtil.parseObj(requestBody.get());
        assertThat(platformTaskId).isEqualTo("agnes-task");
        assertThat(json.getStr("size")).isEqualTo("2K");
        assertThat(json.getStr("ratio")).isEqualTo("16:9");
        verify(imageGenerationService).updateItem(any(ImageItem.class));
        verify(imageGenerationService).update(task);
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
                storageConfigService,
                presetArtStyleResourceResolver
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
                storageConfigService,
                presetArtStyleResourceResolver
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
    void generateUsesLocalPresetArtStyleReferenceWithoutHttpDownload() throws Exception {
        byte[] generatedBytes = "edited-from-preset".getBytes(StandardCharsets.UTF_8);
        String base64Image = Base64.getEncoder().encodeToString(generatedBytes);
        AtomicReference<String> requestPath = new AtomicReference<>();
        AtomicReference<String> requestBody = new AtomicReference<>();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/images/edits", exchange -> {
            requestPath.set(exchange.getRequestURI().getPath());
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.ISO_8859_1));
            writeJson(exchange, 200, """
                    {"data":[{"b64_json":"%s"}],"output_format":"png"}
                    """.formatted(base64Image));
        });
        server.start();

        MediaStorageService mediaStorageService = mock(MediaStorageService.class);
        when(mediaStorageService.storeBytes(any(), eq("images"), eq("png")))
                .thenReturn("/media/images/preset-edited.png");

        OpenAiImageStrategy strategy = new OpenAiImageStrategy(
                mock(ImageGenerationService.class),
                mock(AiModelService.class),
                mock(ModelPresetService.class),
                mediaStorageService,
                storageConfigService,
                presetArtStyleResourceResolver
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .build();

        List<String> urls = strategy.generate(
                "edit with preset",
                "gpt-image-1",
                1024,
                1024,
                1,
                List.of("/art-styles/anime_jp.jpg"),
                apiConfig
        );

        assertThat(urls).containsExactly("/media/images/preset-edited.png");
        assertThat(requestPath.get()).isEqualTo("/v1/images/edits");
        assertThat(requestBody.get()).contains("name=\"image[]\"");
                assertThat(requestBody.get()).contains("filename=\"reference-1.jpg\"");
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

        NewApiImageStrategy strategy = new NewApiImageStrategy(
                mock(ImageGenerationService.class),
                mock(AiModelService.class),
                modelPresetService,
                mock(MediaStorageService.class),
                storageConfigService,
                presetArtStyleResourceResolver
        );
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("newapi")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .build();

        List<String> urls = strategy.generate(
                "a test prompt",
                AiModel.builder()
                        .code("gpt-image-2-official")
                        .capabilityPresetCode("gpt-image-2-official")
                        .build(),
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
