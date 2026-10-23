package com.stonewu.fusion.service.generation.video.strategy.agnes;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategyRouter;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolContext;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import okio.Buffer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class Agnes25VideoProtocolTests {
    private final ModelPresetService presets = new ModelPresetService();
    private final OpenAiCompatibleVideoProtocolSupport support = new OpenAiCompatibleVideoProtocolSupport(
            mock(StorageConfigService.class), new PresetArtStyleResourceResolver());
    private final Agnes25VideoProtocolAdapter adapter = new Agnes25VideoProtocolAdapter(support);

    Agnes25VideoProtocolTests() { presets.init(); }

    @Test
    void sendsNativeRequestAndPollsTheVideoReceiptThroughExplicitProtocol() throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        AtomicReference<JSONObject> request = new AtomicReference<>();
        AtomicReference<String> query = new AtomicReference<>();
        server.createContext("/v1/videos", exchange -> {
            request.set(JSONUtil.parseObj(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)));
            respond(exchange, """
                    {"video_id":"video_25","id":"task_other","task_id":"task_other","status":"queued","seconds":"4"}
                    """);
        });
        server.createContext("/agnesapi", exchange -> {
            query.set(exchange.getRequestURI().getQuery());
            respond(exchange, """
                    {"id":"task_other","status":"completed","internal_status":"pending","seconds":"4",
                     "url":"https://media.example.com/result.mp4","remixed_from_video_id":"video_parent"}
                    """);
        });
        server.start();
        try {
            AiModel model = model(true);
            ApiConfig config = ApiConfig.builder().id(7L).platform("openai_compatible")
                    .apiUrl("http://localhost:" + server.getAddress().getPort() + "/v1")
                    .apiKey("test-key").build();
            ApiConfigService configs = mock(ApiConfigService.class);
            when(configs.getById(7L)).thenReturn(config);
            AiModelMetadataResolver metadata = new AiModelMetadataResolver(configs);
            Agnes25VideoStrategy strategy = new Agnes25VideoStrategy(mock(AiModelService.class), configs,
                    mock(VideoGenerationService.class), presets, metadata, support, adapter);
            VideoGenerationStrategyRouter router = new VideoGenerationStrategyRouter(List.of(strategy), metadata);
            assertThat(router.resolve(model)).isSameAs(strategy);
            VideoTask task = task().duration(4).resolution("720p").count(3).build();
            String receipt = strategy.prepare(task, model, config).submit().providerTaskId();
            assertThat(receipt).isEqualTo("video_25");
            assertThat(request.get().getStr("mode")).isEqualTo("text");
            assertThat(request.get().get("seconds")).isEqualTo("4");
            assertThat(request.get().getStr("size")).isEqualTo("720P");
            assertThat(request.get().getStr("aspect_ratio")).isEqualTo("16:9");
            assertThat(request.get().getInt("n")).isEqualTo(1);
            assertThat(request.get().keySet()).doesNotContain("width", "height", "num_frames", "extra_body", "frame_rate");
            var result = strategy.query(receipt, task, model, config);
            assertThat(query.get()).contains("video_id=video_25", "model_name=agnes-video-2.5-flash");
            assertThat(result.state()).isEqualTo(RemoteGenerationResult.State.SUCCEEDED);
            assertThat(result.outputs().getFirst().media()).isEqualTo("https://media.example.com/result.mp4");
            assertThat(result.outputs().getFirst().durationSeconds()).isEqualTo(4);
        } finally { server.stop(0); }
    }

    @Test
    void mapsFirstAndLastFramesWithoutMergingThemIntoReferences() throws IOException {
        var first = body(task().firstFrameImageUrl("https://media.example.com/first.png").build(), true);
        var last = body(task().lastFrameImageUrl("https://media.example.com/last.png").build(), true);
        var both = body(task().firstFrameImageUrl("https://media.example.com/first.png")
                .lastFrameImageUrl("https://media.example.com/last.png").build(), true);
        assertThat(first.getStr("mode")).isEqualTo("keyframe");
        assertThat(first.getStr("first_frame")).endsWith("first.png");
        assertThat(last.getStr("last_frame")).endsWith("last.png");
        assertThat(both.keySet()).contains("first_frame", "last_frame").doesNotContain("images", "audios");
    }

    @Test
    void supportsAudioOnlyReferencesAndStandardVideoReferences() throws IOException {
        var audio = body(task().referenceAudioUrls("[\"https://media.example.com/track.mp3\"]").build(), true);
        var reference = body(task().referenceImageUrls("[\"https://media.example.com/ref.png\"]")
                .referenceAudioUrls("[\"https://media.example.com/track.mp3\"]")
                .referenceVideoUrls("[\"https://media.example.com/ref.mp4\"]").resolution("2K").build(), false);
        assertThat(audio.getStr("mode")).isEqualTo("reference");
        assertThat(audio.getJSONArray("audios").getStr(0)).endsWith("track.mp3");
        assertThat(reference.getJSONArray("videos").getJSONObject(0).getStr("url")).endsWith("ref.mp4");
        assertThat(reference.getJSONArray("images").getStr(0)).endsWith("ref.png");
        assertThat(reference.getStr("size")).isEqualTo("2K");
    }

    @Test
    void rejectsMixedModesAndUnsupportedInputsBeforeAnyRemoteCall() {
        reject(task().firstFrameImageUrl("https://media.example.com/first.png")
                .referenceAudioUrls("[\"https://media.example.com/track.mp3\"]").build(), "不能同时");
        reject(task().referenceVideoUrls("[\"https://media.example.com/ref.mp4\"]").build(), "不支持参考视频");
        reject(task().referenceImageUrls(JSONUtil.toJsonStr(Collections.nCopies(6, "https://media.example.com/ref.png"))).build(), "数量限制");
        reject(task().referenceAudioUrls(JSONUtil.toJsonStr(Collections.nCopies(4, "https://media.example.com/ref.mp3"))).build(), "3 段");
        reject(task().firstFrameImageUrl("data:image/png;base64,YQ==").build(), "HTTP/HTTPS");
        reject(task().resolution("1080P").build(), "仅支持 720P");
        reject(task().ratio("2:3").build(), "画面比例");
        reject(task().generateAudio(false).build(), "声音开关");
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 3, 13})
    void rejectsOutOfRangeDuration(int seconds) { reject(task().duration(seconds).build(), "4–12"); }

    @Test
    void neverTreatsTaskIdAsVideoReceiptOrRemixIdAsDownloadUrl() {
        var missingReceipt = adapter.parseSubmitResponse(null, "{\"id\":\"task_1\",\"task_id\":\"task_1\",\"status\":\"queued\"}");
        var missingOutput = adapter.parseQueryResponse(null, "{\"status\":\"completed\",\"remixed_from_video_id\":\"video_parent\"}");
        assertThat(missingReceipt.trackingId()).isNull();
        assertThat(missingOutput.videoUrl()).isNull();
    }

    @Test
    void presetsExposeNativeProtocolAndEnforceTheSameCanvasInputRestrictions() {
        var capability = new GenerationModelCapabilityService(new AiModelMetadataResolver(mock(ApiConfigService.class)), presets);
        var flash = model(true);
        assertThat(presets.getPreset(flash.getCode()).getStr("modelProtocol")).isEqualTo("agnes_25");
        assertThat(presets.getPreset("agnes-image-2.5-flash").getJSONObject("config").getJSONObject("supportedSizes")
                .getJSONObject("2K").getStr("16:9")).isEqualTo("2624x1472");
        capability.validateVideoTask(flash, task().referenceAudioUrls("[\"https://media.example.com/track.mp3\"]").build());
        assertThatThrownBy(() -> capability.validateVideoTask(flash,
                task().firstFrameImageUrl("https://media.example.com/first.png")
                        .referenceImageUrls("[\"https://media.example.com/ref.png\"]").build()))
                .isInstanceOf(BusinessException.class).hasMessageContaining("互斥");
    }

    private AiModel model(boolean flash) {
        String code = flash ? "agnes-video-2.5-flash" : "agnes-video-2.5";
        return AiModel.builder().id(8L).apiConfigId(7L).modelType(3).modelProtocol("agnes_25")
                .code(code).capabilityPresetCode(code).build();
    }

    private VideoTask.VideoTaskBuilder task() { return VideoTask.builder().prompt("雨夜灯塔，缓慢推进").duration(5); }

    private JSONObject body(VideoTask task, boolean flash) throws IOException {
        var model = model(flash);
        var context = new OpenAiCompatibleVideoProtocolContext(model, null, task,
                JSONUtil.parseObj(presets.getPresetConfig(model.getCode())), null);
        Buffer buffer = new Buffer();
        adapter.buildSubmitBody(context).writeTo(buffer);
        return JSONUtil.parseObj(buffer.readUtf8());
    }

    private void reject(VideoTask task, String message) {
        assertThatThrownBy(() -> body(task, true)).isInstanceOf(BusinessException.class).hasMessageContaining(message);
    }

    private void respond(HttpExchange exchange, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, bytes.length);
        try (var out = exchange.getResponseBody()) { out.write(bytes); }
    }
}
