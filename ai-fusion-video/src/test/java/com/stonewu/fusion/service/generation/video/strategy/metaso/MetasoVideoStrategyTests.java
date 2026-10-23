package com.stonewu.fusion.service.generation.video.strategy.metaso;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.generation.VideoMediaRules;
import com.stonewu.fusion.service.generation.VideoModelRules;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.video.strategy.minimax.MiniMaxVideoProtocol;
import com.stonewu.fusion.service.media.model.MediaFacts;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.transaction.IllegalTransactionStateException;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class MetasoVideoStrategyTests {
    private static final ModelPresetService PRESETS = new ModelPresetService();
    private final MetasoVideoStrategy strategy = new MetasoVideoStrategy();
    private HttpServer server;

    @BeforeAll
    static void loadPresets() {
        PRESETS.init();
    }

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    private AiModel model(String code) {
        return AiModel.builder().code(code).modelType(3).modelProtocol("metaso_video")
                .capabilityPresetCode("metaso:" + code)
                .config(PRESETS.getPresetConfig("metaso:" + code)).build();
    }

    private VideoTask task() {
        return VideoTask.builder().prompt("一艘小船驶向灯塔").resolution("768P").ratio("16:9").duration(5).build();
    }

    /** 保护秘塔自己的时长/模式合同，避免误用官方 H3 或其他 Seedance 渠道预设。 */
    @Test
    void platformLimitsDependOnResolutionAndInputMode() {
        var h3 = JSONUtil.parseObj(model("MiniMax-H3").getConfig());
        var task = task();
        task.setDuration(30);
        VideoModelRules.normalizeTask(h3, task);
        task.setResolution("2K");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(h3, task)).hasMessageContaining("最长支持 15 秒");
        task.setDuration(15);
        task.setFirstFrameImageUrl("https://example.com/first.jpg");
        task.setReferenceImageUrls("[\"https://example.com/character.jpg\"]");
        assertThatCode(() -> VideoModelRules.normalizeTask(h3, task)).doesNotThrowAnyException();
        task.setReferenceVideoUrls("[\"https://example.com/reference.mp4\"]");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(h3, task)).hasMessageContaining("不能与参考视频混用");
        var seedance = JSONUtil.parseObj(model("seedance-2.0-mini").getConfig());
        task.setReferenceVideoUrls(null);
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(seedance, task)).hasMessageContaining("不能混用");
        task.setFirstFrameImageUrl(null);
        task.setReferenceImageUrls(null);
        task.setResolution("1080p");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(seedance, task)).hasMessageContaining("分辨率");
    }

    /** 单视频的 15.5 秒容差不能放宽多个视频合计 15 秒的实际限制。 */
    @Test
    void singleReferenceToleranceDoesNotIncreaseTheMultipleVideoBudget() {
        var config = JSONUtil.parseObj(model("MiniMax-H3").getConfig());
        assertThatCode(() -> VideoMediaRules.validate(config, 5, List.of(), List.of(video(15500)), List.of()))
                .doesNotThrowAnyException();
        assertThatThrownBy(() -> VideoMediaRules.validate(config, 5, List.of(),
                List.of(video(8000), video(7100)), List.of())).hasMessageContaining("总时长不能超过 15.0");
        assertThatThrownBy(() -> VideoMediaRules.validate(config, 5, List.of(), List.of(video(15501)), List.of()))
                .hasMessageContaining("每段时长");
    }

    private MediaFacts video(long milliseconds) {
        return new MediaFacts(MediaKind.VIDEO, "video/mp4", 4096, "test-hash", 1280, 720, milliseconds, true);
    }

    /** HTTP 合同验证保留真实 model ID、鉴权头、V2 路径以及非终态/终态区分。 */
    @Test
    void submitsAndQueriesTheDocumentedV2Contract() throws Exception {
        var observed = new CopyOnWriteArrayList<Observed>();
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/api/minimax/v2", exchange -> {
            observed.add(new Observed(exchange.getRequestMethod(), exchange.getRequestURI().toString(),
                    exchange.getRequestHeaders().getFirst("Authorization"),
                    new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)));
            String response = exchange.getRequestMethod().equals("POST") ? "{\"task_id\":\"task-123\"}"
                    : "{\"task\":{\"status\":\"succeeded\",\"duration\":5,\"content\":{\"url\":\"https://example.com/output.mp4\"}}}";
            byte[] bytes = response.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        var api = ApiConfig.builder().apiUrl("http://127.0.0.1:" + server.getAddress().getPort() + "/api/minimax")
                .apiKey("local-contract-key").build();
        var model = model("MiniMax-H3");
        var task = task();
        var call = strategy.prepare(task, model, api);
        var receipt = call.submit();
        assertThat(receipt.providerTaskId()).isEqualTo("task-123");
        var result = strategy.query(receipt.providerTaskId(), task, model, api);
        assertThat(result.state()).isEqualTo(RemoteGenerationResult.State.SUCCEEDED);
        assertThat(result.outputs().getFirst().media()).isEqualTo("https://example.com/output.mp4");
        assertThat(observed).hasSize(2);
        assertThat(observed.getFirst().path()).isEqualTo("/api/minimax/v2/video_generation");
        assertThat(observed.getLast().path()).isEqualTo("/api/minimax/v2/query/video_generation/task-123");
        assertThat(observed).allMatch(request -> request.authorization().equals("Bearer local-contract-key"));
        var body = JSONUtil.parseObj(observed.getFirst().body());
        assertThat(body.getStr("model")).isEqualTo("MiniMax-H3");
        assertThat(body.getBool("fast_mode")).isFalse();
        assertThat(body.getBool("context_ir_enabled")).isFalse();
        assertThat(body.toString()).doesNotContain("local-contract-key");

        TransactionSynchronizationManager.setActualTransactionActive(true);
        try {
            assertThatThrownBy(() -> strategy.query("task-123", task, model, api))
                    .isInstanceOf(IllegalTransactionStateException.class);
        } finally {
            TransactionSynchronizationManager.setActualTransactionActive(false);
        }
        assertThat(observed).hasSize(2);
    }

    @Test
    void optionalFeaturesAreExplicitAndSeedanceRejectsH3OnlyOptions() {
        var h3 = model("MiniMax-H3");
        var options = JSONUtil.parseObj(h3.getConfig()).set("contextIrEnabled", true).set("fastMode", true);
        h3.setConfig(options.toString());
        var body = strategy.buildRequest(task(), h3);
        assertThat(body.getBool("context_ir_enabled")).isTrue();
        assertThat(body.getBool("fast_mode")).isTrue();
        var seedance = model("seedance-2.0");
        assertThat(strategy.buildRequest(task(), seedance).containsKey("fast_mode")).isFalse();
        seedance.setConfig(JSONUtil.parseObj(seedance.getConfig()).set("fastMode", true).toString());
        assertThatThrownBy(() -> strategy.buildRequest(task(), seedance)).hasMessageContaining("不支持");
    }

    @Test
    void pendingFailedAndMalformedResultsCannotBePublishedAsSuccess() {
        assertThat(MiniMaxVideoProtocol.result(JSONUtil.parseObj("{\"task\":{\"status\":\"running\"}}"))
                .state()).isEqualTo(RemoteGenerationResult.State.PENDING);
        var failed = MiniMaxVideoProtocol.result(JSONUtil.parseObj(
                "{\"task\":{\"status\":\"failed\",\"error\":{\"code\":\"bad_input\",\"message\":\"invalid image\"}}}"));
        assertThat(failed.state()).isEqualTo(RemoteGenerationResult.State.FAILED);
        assertThat(failed.error()).contains("invalid image");
        assertThatThrownBy(() -> MiniMaxVideoProtocol.result(new JSONObject())).hasMessageContaining("缺少任务");
        assertThatThrownBy(() -> MiniMaxVideoProtocol.result(JSONUtil.parseObj(
                "{\"task\":{\"status\":\"succeeded\"}}"))).hasMessageContaining("没有返回视频地址");
    }

    private record Observed(String method, String path, String authorization, String body) {
    }
}
