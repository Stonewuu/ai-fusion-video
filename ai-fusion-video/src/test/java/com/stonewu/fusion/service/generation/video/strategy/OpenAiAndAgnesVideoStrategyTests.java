package com.stonewu.fusion.service.generation.video.strategy;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.agnes.AgnesVideoStrategy;
import com.stonewu.fusion.service.generation.video.strategy.agnes.AgnesVideoProtocolAdapter;
import com.stonewu.fusion.service.generation.video.strategy.openai.OpenAiVideoProtocolAdapter;
import com.stonewu.fusion.service.generation.video.strategy.openai.OpenAiVideoStrategy;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import com.stonewu.fusion.service.storage.MediaStorageService;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OpenAiAndAgnesVideoStrategyTests {

    private HttpServer server;

    private final StorageConfigService storageConfigService = mock(StorageConfigService.class);
    private final PresetArtStyleResourceResolver presetArtStyleResourceResolver = new PresetArtStyleResourceResolver();

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void getNameUsesOfficialOpenAiProtocolKey() {
        OpenAiVideoStrategy strategy = newOpenAiStrategy(
            mock(AiModelService.class),
            mock(ApiConfigService.class),
            mock(VideoGenerationService.class),
            mock(ModelPresetService.class),
            mock(MediaStorageService.class)
        );

        assertThat(strategy.getName()).isEqualTo("openai");
    }

    @Test
    void openAiAdapterParsesOnlyOfficialVideoJobFields() {
        OpenAiCompatibleVideoProtocolSupport support = new OpenAiCompatibleVideoProtocolSupport(
                storageConfigService,
                presetArtStyleResourceResolver
        );
        OpenAiVideoProtocolAdapter adapter = new OpenAiVideoProtocolAdapter(support);

        var official = adapter.parseSubmitResponse(null, """
                {"id":"video_123","status":"queued","seconds":"8"}
                """);
        var compatibleOnly = adapter.parseSubmitResponse(null, """
                {"task_id":"task_123","state":"queued","duration":"8","video_url":"https://example.com/video.mp4"}
                """);

        assertThat(adapter.getProtocol()).isEqualTo("openai");
        assertThat(official.trackingId()).isEqualTo("video_123");
        assertThat(official.status()).isEqualTo("queued");
        assertThat(official.durationSeconds()).isEqualTo(8);
        assertThat(compatibleOnly.trackingId()).isNull();
        assertThat(compatibleOnly.status()).isNull();
        assertThat(compatibleOnly.videoUrl()).isNull();
    }

    @Test
    void submitAndPollText2VideoStoresLocalVideoContent() throws Exception {
        byte[] videoBytes = "fake-mp4-bytes".getBytes(StandardCharsets.UTF_8);
        byte[] coverBytes = "fake-cover-bytes".getBytes(StandardCharsets.UTF_8);
        AtomicReference<String> submitPath = new AtomicReference<>();
        AtomicReference<String> submitBody = new AtomicReference<>();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/videos", exchange -> {
            String path = exchange.getRequestURI().getPath();
            String query = exchange.getRequestURI().getQuery();
            if ("POST".equalsIgnoreCase(exchange.getRequestMethod()) && "/v1/videos".equals(path)) {
                submitPath.set(path);
                submitBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
                writeJson(exchange, """
                        {"id":"video_123","object":"video","model":"sora-2","status":"queued","progress":0}
                        """);
                return;
            }
            if ("/v1/videos/video_123/content".equals(path)) {
                byte[] payload = "thumbnail".equals(query == null ? "" : query.replace("variant=", ""))
                        ? coverBytes : videoBytes;
                String contentType = payload == coverBytes ? "image/jpeg" : "video/mp4";
                writeBinary(exchange, payload, contentType);
                return;
            }
            if ("/v1/videos/video_123".equals(path)) {
                writeJson(exchange, """
                        {"id":"video_123","status":"completed","progress":100,"seconds":"4"}
                        """);
                return;
            }
            exchange.sendResponseHeaders(404, -1);
            exchange.close();
        });
        server.start();

        AiModel model = AiModel.builder().id(10L).code("sora-2").apiConfigId(1L).build();
        ApiConfig apiConfig = ApiConfig.builder()
                .platform("openai_compatible")
                .apiUrl("http://localhost:" + server.getAddress().getPort())
                .apiKey("test-key")
                .build();

        AiModelService aiModelService = mock(AiModelService.class);
        when(aiModelService.getById(10L)).thenReturn(model);
        ApiConfigService apiConfigService = mock(ApiConfigService.class);
        when(apiConfigService.getById(1L)).thenReturn(apiConfig);

        VideoItem item = VideoItem.builder().id(100L).taskId(5L).status(0).build();
        VideoGenerationService videoGenerationService = mock(VideoGenerationService.class);
        when(videoGenerationService.listItems(5L)).thenReturn(List.of(item));

        MediaStorageService mediaStorageService = mock(MediaStorageService.class);
        when(mediaStorageService.storeBytes(any(), eq("videos"), eq("mp4")))
                .thenReturn("/media/videos/result.mp4");
        when(mediaStorageService.storeBytes(any(), eq("images"), eq("jpg")))
                .thenReturn("/media/images/cover.jpg");

        OpenAiVideoStrategy strategy = newOpenAiStrategy(
            aiModelService,
            apiConfigService,
            videoGenerationService,
            mock(ModelPresetService.class),
            mediaStorageService
        );

        VideoTask task = VideoTask.builder().id(5L).taskId("task-1").modelId(10L).prompt("a cat playing piano")
                .duration(4).resolution("1280x720").build();

        String platformTaskId = strategy.submit(task);
        assertThat(platformTaskId).isEqualTo("video_123");
        assertThat(item.getPlatformTaskId()).isEqualTo("video_123");
        assertThat(submitPath.get()).isEqualTo("/v1/videos");
        assertThat(submitBody.get()).contains("sora-2");
        assertThat(submitBody.get()).contains("a cat playing piano");
        assertThat(submitBody.get()).contains("1280x720");

        strategy.poll(platformTaskId, task);

        assertThat(item.getVideoUrl()).isEqualTo("/media/videos/result.mp4");
        assertThat(item.getCoverUrl()).isEqualTo("/media/images/cover.jpg");
        assertThat(item.getStatus()).isEqualTo(1);
        assertThat(item.getDuration()).isEqualTo(4);
        verify(mediaStorageService).storeBytes(videoBytes, "videos", "mp4");
    }

        @Test
        void submitAndPollAgnesProtocolUsesJsonBodyAndVideoIdQuery() throws Exception {
        AtomicReference<String> submitBody = new AtomicReference<>();
        AtomicReference<String> queryPath = new AtomicReference<>();
        AtomicReference<String> queryString = new AtomicReference<>();

        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/videos", exchange -> {
            if ("POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            submitBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            writeJson(exchange, """
                {"id":"task_123","task_id":"task_123","video_id":"video_abc","status":"queued","seconds":"5.0"}
                """);
            return;
            }
            exchange.sendResponseHeaders(404, -1);
            exchange.close();
        });
        server.createContext("/agnesapi", exchange -> {
            queryPath.set(exchange.getRequestURI().getPath());
            queryString.set(exchange.getRequestURI().getQuery());
            writeJson(exchange, """
                {"id":"task_123","video_id":"video_abc","status":"completed","seconds":"5.0","remixed_from_video_id":"https://cdn.example.com/result.mp4","error":null}
                """);
        });
        server.start();

        AiModel model = AiModel.builder()
            .id(11L)
            .code("agnes-video-v2.0")
            .apiConfigId(2L)
            .capabilityPresetCode("agnes-video-v2.0")
            .modelProtocol("agnes")
            .config("{\"frame_rate\":24}")
            .build();
        ApiConfig apiConfig = ApiConfig.builder()
            .platform("openai_compatible")
            .apiUrl("http://localhost:" + server.getAddress().getPort())
            .apiKey("test-key")
            .build();

        AiModelService aiModelService = mock(AiModelService.class);
        when(aiModelService.getById(11L)).thenReturn(model);
        ApiConfigService apiConfigService = mock(ApiConfigService.class);
        when(apiConfigService.getById(2L)).thenReturn(apiConfig);

        VideoItem item = VideoItem.builder().id(101L).taskId(6L).status(0).build();
        VideoGenerationService videoGenerationService = mock(VideoGenerationService.class);
        when(videoGenerationService.listItems(6L)).thenReturn(List.of(item));

        AgnesVideoStrategy strategy = newAgnesStrategy(
            aiModelService,
            apiConfigService,
            videoGenerationService,
            mock(ModelPresetService.class),
            mock(MediaStorageService.class)
        );

        VideoTask task = VideoTask.builder()
            .id(6L)
            .taskId("task-agnes")
            .modelId(11L)
            .prompt("a cinematic ocean wave")
            .duration(5)
            .firstFrameImageUrl("https://example.com/first.png")
            .lastFrameImageUrl("https://example.com/last.png")
            .build();

        String trackingId = strategy.submit(task);
        assertThat(trackingId).isEqualTo("video_abc");
        assertThat(item.getPlatformTaskId()).isEqualTo("video_abc");
        assertThat(submitBody.get()).contains("\"model\":\"agnes-video-v2.0\"");
        assertThat(submitBody.get()).contains("\"prompt\":\"a cinematic ocean wave\"");
        assertThat(submitBody.get()).contains("\"frame_rate\":24");
        assertThat(submitBody.get()).contains("\"num_frames\":121");
        assertThat(submitBody.get()).contains("\"mode\":\"keyframes\"");
        assertThat(submitBody.get()).contains("https://example.com/first.png");
        assertThat(submitBody.get()).contains("https://example.com/last.png");

        strategy.poll(trackingId, task);

        assertThat(queryPath.get()).isEqualTo("/agnesapi");
        assertThat(queryString.get()).contains("video_id=video_abc");
        assertThat(queryString.get()).contains("model_name=agnes-video-v2.0");
        assertThat(item.getVideoUrl()).isEqualTo("https://cdn.example.com/result.mp4");
        assertThat(item.getStatus()).isEqualTo(1);
        assertThat(item.getDuration()).isEqualTo(5);
        }

    private OpenAiVideoStrategy newOpenAiStrategy(AiModelService aiModelService,
                                                  ApiConfigService apiConfigService,
                                                  VideoGenerationService videoGenerationService,
                                                  ModelPresetService modelPresetService,
                                                  MediaStorageService mediaStorageService) {
        OpenAiCompatibleVideoProtocolSupport support = new OpenAiCompatibleVideoProtocolSupport(
                storageConfigService,
                presetArtStyleResourceResolver
        );
        return new OpenAiVideoStrategy(
                aiModelService,
                apiConfigService,
                videoGenerationService,
                modelPresetService,
                mediaStorageService,
                new AiModelMetadataResolver(apiConfigService),
                support,
                new OpenAiVideoProtocolAdapter(support)
        );
    }

    private AgnesVideoStrategy newAgnesStrategy(AiModelService aiModelService,
                                                ApiConfigService apiConfigService,
                                                VideoGenerationService videoGenerationService,
                                                ModelPresetService modelPresetService,
                                                MediaStorageService mediaStorageService) {
        OpenAiCompatibleVideoProtocolSupport support = new OpenAiCompatibleVideoProtocolSupport(
                storageConfigService,
                presetArtStyleResourceResolver
        );
        return new AgnesVideoStrategy(
                aiModelService,
                apiConfigService,
                videoGenerationService,
                modelPresetService,
                mediaStorageService,
                new AiModelMetadataResolver(apiConfigService),
                support,
                new AgnesVideoProtocolAdapter(support)
        );
    }

    private static void writeJson(HttpExchange exchange, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static void writeBinary(HttpExchange exchange, byte[] payload, String contentType) throws IOException {
        exchange.getResponseHeaders().add("Content-Type", contentType);
        exchange.sendResponseHeaders(200, payload.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(payload);
        }
    }
}
