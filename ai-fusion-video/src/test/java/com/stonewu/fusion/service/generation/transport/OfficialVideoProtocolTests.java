package com.stonewu.fusion.service.generation.transport;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.generation.VideoModelRules;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import com.stonewu.fusion.service.generation.video.strategy.dashscope.Wan30VideoStrategy;
import com.stonewu.fusion.service.generation.video.strategy.minimax.MiniMaxVideoStrategy;
import com.stonewu.fusion.service.generation.video.strategy.volcengine.VolcengineVideoStrategy;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.*;

class OfficialVideoProtocolTests {
    private static final ModelPresetService PRESETS = new ModelPresetService();
    private HttpServer server;
    private ApiConfig api;
    private String response;
    private int status;
    private final List<Received> requests = new CopyOnWriteArrayList<>();

    record Received(String method, String path, String auth, String async, JSONObject body) {
    }

    record Protocol(String code, VideoGenerationStrategy strategy, String version, String create, String query,
                    String receipt) {
    }

    @BeforeAll
    static void load() {
        PRESETS.init();
    }

    static Stream<Protocol> protocols() {
        return Stream.of(
                new Protocol("doubao-seedance-2-5-260628", new VolcengineVideoStrategy(), "/api/v3",
                        "/api/v3/contents/generations/tasks", "/api/v3/contents/generations/tasks",
                        "{\"id\":\"task-1\"}"),
                new Protocol("MiniMax-H3-Max", new MiniMaxVideoStrategy(), "/v2", "/v2/video_generation",
                        "/v2/query/video_generation", "{\"task_id\":\"task-1\"}"),
                new Protocol("wan3.0-video", new Wan30VideoStrategy(), "/api/v1",
                        "/api/v1/services/aigc/video-generation/video-synthesis", "/api/v1/tasks",
                        "{\"output\":{\"task_id\":\"task-1\"}}"));
    }

    @BeforeEach
    void start() throws IOException {
        status = 200;
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", this::handle);
        server.start();
        api = ApiConfig.builder().apiUrl("http://127.0.0.1:" + server.getAddress().getPort())
                .apiKey("contract-test-only").build();
    }

    @AfterEach
    void stop() {
        server.stop(0);
    }

    private void handle(HttpExchange exchange) throws IOException {
        String raw = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        requests.add(new Received(exchange.getRequestMethod(), exchange.getRequestURI().getRawPath(),
                exchange.getRequestHeaders().getFirst("Authorization"),
                exchange.getRequestHeaders().getFirst("X-DashScope-Async"),
                raw.isBlank() ? new JSONObject() : JSONUtil.parseObj(raw)));
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.getResponseHeaders().add("Retry-After", "0");
        byte[] body = response.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(status, body.length);
        exchange.getResponseBody().write(body);
        exchange.close();
    }

    private AiModel model(Protocol protocol) {
        return AiModel.builder().code(protocol.code()).config(PRESETS.getPresetConfig(protocol.code())).build();
    }

    private VideoTask task(AiModel model) {
        var task = VideoTask.builder().prompt("雨夜灯塔").referenceImageUrls("[\"https://example.com/image.png\"]")
                .referenceVideoUrls("[\"https://example.com/video.mp4\"]")
                .referenceAudioUrls("[\"https://example.com/audio.mp3\"]").build();
        VideoModelRules.normalizeTask(JSONUtil.parseObj(model.getConfig()), task);
        return task;
    }

    @ParameterizedTest
    @MethodSource("protocols")
    void nativeProtocolsKeepResolutionAndRatioAsSeparateFields(Protocol protocol) {
        var model = model(protocol);
        String resolution = protocol.strategy() instanceof MiniMaxVideoStrategy ? "768P"
                : protocol.strategy() instanceof Wan30VideoStrategy ? "720P" : "720p";
        var task = VideoTask.builder().prompt("横屏海景").resolution(resolution).ratio("16:9").duration(5).build();
        VideoModelRules.normalizeTask(JSONUtil.parseObj(model.getConfig()), task);
        response = protocol.receipt();
        try (var prepared = protocol.strategy().prepare(task, model, api)) {
            prepared.submit();
        }
        var body = requests.getFirst().body();
        var parameters = protocol.strategy() instanceof Wan30VideoStrategy ? body.getJSONObject("parameters") : body;
        assertThat(parameters.getStr("resolution")).isEqualTo(resolution);
        assertThat(parameters.getStr("ratio")).isEqualTo("16:9");
        assertThat(parameters.keySet()).doesNotContain("size");
    }

    @ParameterizedTest
    @MethodSource("protocols")
    void submitsExactlyOneRequestUsingOfficialEndpointAndFields(Protocol protocol) {
        var model = model(protocol);
        var task = task(model);
        response = protocol.receipt();
        try (var prepared = protocol.strategy().prepare(task, model, api)) {
            assertThat(requests).isEmpty();
            assertThat(prepared.submit().providerTaskId()).isEqualTo("task-1");
            assertThatThrownBy(prepared::submit).isInstanceOf(IllegalStateException.class);
        }
        assertThat(requests).hasSize(1);
        var sent = requests.getFirst();
        assertThat(sent.method()).isEqualTo("POST");
        assertThat(sent.path()).isEqualTo(protocol.create());
        assertThat(sent.auth()).isEqualTo("Bearer contract-test-only");
        assertThat(sent.body().getStr("model")).isEqualTo(protocol.code());
        if (protocol.strategy() instanceof Wan30VideoStrategy) {
            assertThat(sent.async()).isEqualTo("enable");
            var media = sent.body().getJSONObject("input").getJSONArray("media");
            assertThat(media.getJSONObject(2).getStr("type")).isEqualTo("reference_audio");
            assertThat(sent.body().getJSONObject("parameters").getBool("audio")).isTrue();
            assertThat(sent.body().getJSONObject("parameters").getBool("prompt_extend")).isTrue();
        } else {
            var content = sent.body().getJSONArray("content");
            assertThat(content.getJSONObject(3).getStr("type")).isEqualTo("audio_url");
            assertThat(content.getJSONObject(3).getStr("role")).isEqualTo("reference_audio");
            if (protocol.strategy() instanceof MiniMaxVideoStrategy) {
                assertThat(sent.body().keySet()).doesNotContain("generate_audio", "watermark", "seed", "camera_fixed");
                assertThat(sent.body().getJSONObject("extra").getStr("prompt_expansion_mode")).isEqualTo("balanced");
            } else {
                assertThat(sent.body().getInt("duration")).isEqualTo(5);
                assertThat(sent.body().getStr("omni_reference_task_type")).isEqualTo("auto");
                assertThat(sent.body().getStr("output_format")).isEqualTo("mp4");
                assertThat(sent.body().keySet()).doesNotContain("camera_fixed");
            }
        }
    }

    @ParameterizedTest
    @MethodSource("protocols")
    void versionedBaseUrlIsNotDuplicatedAndQueryTracksSameId(Protocol protocol) {
        var model = model(protocol);
        var task = task(model);
        api.setApiUrl(api.getApiUrl() + protocol.version() + "/");
        response = query(protocol, "succeeded");
        var result = protocol.strategy().query("task/with space", task, model, api);
        assertThat(result.state()).isEqualTo(RemoteGenerationResult.State.SUCCEEDED);
        assertThat(result.outputs().getFirst().durationSeconds()).isEqualTo(6);
        assertThat(result.outputs().getFirst().media()).isEqualTo("https://example.com/result.mp4");
        assertThat(requests.getFirst().path()).isEqualTo(protocol.query() + "/task%2Fwith%20space");
        assertThat(requests.getFirst().method()).isEqualTo("GET");
        response = query(protocol, "running");
        assertThat(protocol.strategy().query("task-1", task, model, api).state()).isEqualTo(
                RemoteGenerationResult.State.PENDING);
        response = query(protocol, "failed");
        var failure = protocol.strategy().query("task-1", task, model, api);
        assertThat(failure.state()).isEqualTo(RemoteGenerationResult.State.FAILED);
        assertThat(failure.error()).contains("InvalidInput", "参考视频过长");
        response = query(protocol, "cancelled");
        assertThat(protocol.strategy().query("task-1", task, model, api).state()).isEqualTo(
                RemoteGenerationResult.State.CANCELLED);
    }

    @ParameterizedTest
    @MethodSource("protocols")
    void explicit503IsFailureWithReasonAndNeverResubmitted(Protocol protocol) {
        status = 503;
        response = "{\"error\":{\"code\":\"QueueFull\",\"message\":\"video queue is full\"}}";
        var model = model(protocol);
        try (var call = protocol.strategy().prepare(task(model), model, api)) {
            var result = call.submit().result();
            assertThat(result.state()).isEqualTo(RemoteGenerationResult.State.FAILED);
            assertThat(result.error()).contains("503", "video queue is full");
        }
        assertThat(requests).hasSize(1);
    }

    private String query(Protocol p, String state) {
        if (p.strategy() instanceof Wan30VideoStrategy) {
            return JSONUtil.createObj()
                    .set("output", new JSONObject().set("task_status", state)
                            .set("video_url", "https://example.com/result.mp4").set("code", "InvalidInput")
                            .set("message", "参考视频过长"))
                    .set("usage", new JSONObject().set("duration", 21).set("output_video_duration", 6)).toString();
        }
        var result = new JSONObject().set("status", state).set("duration", 6)
                .set("content", new JSONObject().set(p.strategy() instanceof MiniMaxVideoStrategy ? "url" : "video_url",
                        "https://example.com/result.mp4"))
                .set("error", new JSONObject().set("code", "InvalidInput").set("message", "参考视频过长"));
        return p.strategy() instanceof MiniMaxVideoStrategy ? new JSONObject().set("task", result)
                .toString() : result.toString();
    }
}
