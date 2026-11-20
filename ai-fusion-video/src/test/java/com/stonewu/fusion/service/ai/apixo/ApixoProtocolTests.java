package com.stonewu.fusion.service.ai.apixo;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.catalog.ApixoModelDiscovery;
import com.stonewu.fusion.service.ai.catalog.ProviderCatalogService;
import com.stonewu.fusion.service.generation.VideoModelRules;
import com.stonewu.fusion.service.generation.image.strategy.apixo.ApixoImageStrategy;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.video.strategy.apixo.ApixoVideoStrategy;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.transaction.support.TransactionSynchronizationManager;
import org.springframework.transaction.IllegalTransactionStateException;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Collections;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 验证异步任务、分辨率与素材组合、目录鉴权和只提交一次的 HTTP 行为。 */
class ApixoProtocolTests {
    private static final ModelPresetService PRESETS = new ModelPresetService();
    private final CopyOnWriteArrayList<Call> calls = new CopyOnWriteArrayList<>();
    private HttpServer server;
    private ApiConfig api;
    private String state = "pending";
    private String returnedId = "apixo-task-1";
    private String result = "{\"resultUrls\":[\"https://cdn.example.com/result.png\"]}";
    private int code = 200;
    private boolean disconnect;

    record Call(String method, String path, String query, String authorization, String body) {
    }

    @BeforeAll
    static void presets() {
        PRESETS.init();
    }

    @BeforeEach
    void start() throws Exception {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            String path = exchange.getRequestURI().getPath();
            calls.add(new Call(exchange.getRequestMethod(), path, exchange.getRequestURI().getQuery(),
                    exchange.getRequestHeaders().getFirst("Authorization"),
                    new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)));
            if (disconnect) {
                exchange.close();
                return;
            }
            var data = new JSONObject().set("taskId", returnedId).set("state", state)
                    .set("resultJson", result).set("failCode", "CONTENT_VIOLATION");
            var body = new JSONObject().set("code", code).set("message", "fixture")
                    .set("data", path.endsWith("current-balance") ? 0 : data);
            byte[] bytes = body.toString().getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        api = ApiConfig.builder().providerId("apixo").platform("openai_compatible")
                .apiUrl("http://127.0.0.1:" + server.getAddress().getPort()).apiKey("fixture-api-key")
                .autoAppendV1Path(false).build();
    }

    @AfterEach
    void stop() {
        TransactionSynchronizationManager.setActualTransactionActive(false);
        server.stop(0);
    }

    @Test
    void imageIntentPrecedesSingleSubmissionAndSuccessRequiresCompletedFiles() {
        var strategy = new ApixoImageStrategy();
        var model = model("seedream-5-5");
        var task = ImageTask.builder().prompt("白色背景上的蓝色立方体").resolution("1.5K").ratio("16:9").build();
        try (var prepared = strategy.prepare(task, model, api)) {
            assertThat(calls).isEmpty();
            assertThat(prepared.intent().payload().toString()).contains("seedream-5-5", "async")
                    .doesNotContain("fixture-api-key", "Authorization");
            assertThat(prepared.submit().providerTaskId()).isEqualTo("apixo-task-1");
            assertThatThrownBy(prepared::submit).hasMessageContaining("不能再次调用");
        }
        var submitted = JSONUtil.parseObj(calls.getFirst().body());
        assertThat(submitted.getStr("request_type")).isEqualTo("async");
        assertThat(submitted.getJSONObject("input").getStr("mode")).isEqualTo("text-to-image");
        assertThat(submitted.getJSONObject("input").getStr("resolution")).isEqualTo("1.5k");
        assertThat(strategy.query("apixo-task-1", task, model, api).state()).isEqualTo(RemoteGenerationResult.State.PENDING);
        state = "processing";
        assertThat(strategy.query("apixo-task-1", task, model, api).state()).isEqualTo(RemoteGenerationResult.State.PENDING);
        state = "success";
        assertThat(strategy.query("apixo-task-1", task, model, api).outputs().getFirst().media())
                .isEqualTo("https://cdn.example.com/result.png");
        assertThat(calls).filteredOn(call -> call.method().equals("POST")).hasSize(1);
        assertThat(calls.getFirst().path()).isEqualTo("/api/v1/generateTask/seedream-5-5");
        assertThat(calls.getLast().path()).isEqualTo("/api/v1/statusTask/seedream-5-5");
        assertThat(calls.getLast().query()).isEqualTo("taskId=apixo-task-1");
        assertThat(calls).extracting(Call::authorization).containsOnly("Bearer fixture-api-key");
    }

    @ParameterizedTest
    @ValueSource(strings = {"seedream-5-5", "seedream-5-0-pro", "seedream-5-0", "seedream-4-5",
            "nano-banana-2", "nano-banana-pro", "nano-banana", "gpt-image-2", "flux-2"})
    void imageCatalogCanSubmitDefaultsAndReferenceEditing(String name) {
        var task = ImageTask.builder().prompt("白色背景上的蓝色立方体").build();
        var plain = ApixoGenerationRequests.image(task, model(name));
        assertThat(plain.getStr("mode")).isEqualTo(name.equals("flux-2") ? "pro-text-to-image" : "text-to-image");
        task.setRefImageUrls("[\"https://example.com/reference.png\"]");
        var editing = ApixoGenerationRequests.image(task, model(name));
        assertThat(editing.getStr("mode")).isEqualTo(name.equals("flux-2") ? "pro-image-to-image" : "image-to-image");
        assertThat(editing.getJSONArray("image_urls").toList(String.class)).containsExactly("https://example.com/reference.png");
        if (name.equals("seedream-4-5") || name.equals("seedream-5-0")) {
            assertThat(editing.getInt("max_images")).isEqualTo(1);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"seedance-2-5", "seedance-2-5-flash", "seedance-2-0", "seedance-2-0-fast",
            "seedance-2-0-mini", "minimax-h3", "minimax-h3-LoRA", "sora-2", "veo-3-1"})
    void videoCatalogDefaultsSurviveSharedValidationAndUseNativeModes(String name) {
        var model = model(name);
        var task = VideoTask.builder().prompt("固定镜头，蓝色立方体缓慢旋转").build();
        VideoModelRules.normalizeTask(JSONUtil.parseObj(model.getConfig()), task);
        var body = ApixoGenerationRequests.video(task, model);
        assertThat(body.getStr(name.equals("veo-3-1") ? "generationType" : "mode"))
                .isEqualTo(name.equals("veo-3-1") ? "TEXT_2_VIDEO" : "text-to-video");
        assertThat(body.getInt("duration")).isBetween(4, 30);
        if (name.equals("sora-2")) {
            assertThat(body.containsKey("resolution")).isFalse();
            assertThat(body.getStr("aspect_ratio")).isEqualTo("landscape");
        }
    }

    @Test
    void seedancePreservesFrameOrderAndSeparatesOmniReferences() {
        var model = model("seedance-2-5");
        var task = VideoTask.builder().prompt("从室内移动到花园").duration(-1).resolution("720p").ratio("adaptive")
                .firstFrameImageUrl("https://example.com/first.png").lastFrameImageUrl("https://example.com/last.png").build();
        var frames = ApixoGenerationRequests.video(task, model);
        assertThat(frames.getStr("mode")).isEqualTo("first_and_last_frames");
        assertThat(frames.getStr("aspect_ratio")).isEqualTo("auto");
        assertThat(frames.getInt("duration")).isEqualTo(-1);
        assertThat(frames.getJSONArray("image_urls").toList(String.class))
                .containsExactly("https://example.com/first.png", "https://example.com/last.png");
        task.setReferenceVideoUrls("[\"https://example.com/reference.mp4\"]");
        assertThatThrownBy(() -> ApixoGenerationRequests.video(task, model)).hasMessageContaining("分开使用");
        task.setFirstFrameImageUrl(null);
        task.setLastFrameImageUrl(null);
        task.setReferenceAudioUrls("[\"https://example.com/voice.wav\"]");
        var omni = ApixoGenerationRequests.video(task, model);
        assertThat(omni.getStr("mode")).isEqualTo("omni_reference");
        assertThat(omni.getJSONArray("video_urls")).hasSize(1);
        assertThat(omni.getJSONArray("audio_urls")).hasSize(1);
    }

    @ParameterizedTest
    @ValueSource(strings = {"seedance-2-5", "seedance-2-5-flash"})
    void seedance25FramesRequireAutomaticRatioBeforeSubmission(String name) {
        var model = model(name);
        var config = JSONUtil.parseObj(model.getConfig());
        var task = VideoTask.builder().prompt("陶瓷杯缓慢旋转").duration(4).resolution("480p").ratio("16:9")
                .firstFrameImageUrl("https://example.com/first.png").build();
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config, task)).hasMessageContaining("自适应");
        task.setRatio("adaptive");
        VideoModelRules.normalizeTask(config, task);
        assertThat(ApixoGenerationRequests.video(task, model).getStr("aspect_ratio")).isEqualTo("auto");
        task.setLastFrameImageUrl("https://example.com/last.png");
        VideoModelRules.normalizeTask(config, task);
        assertThat(ApixoGenerationRequests.video(task, model).getJSONArray("image_urls").toList(String.class))
                .containsExactly("https://example.com/first.png", "https://example.com/last.png");
        task.setFirstFrameImageUrl(null);
        task.setLastFrameImageUrl(null);
        task.setRatio("16:9");
        VideoModelRules.normalizeTask(config, task);
        assertThat(ApixoGenerationRequests.video(task, model).getStr("aspect_ratio")).isEqualTo("16:9");
        assertThat(calls).isEmpty();
    }

    @Test
    void invalidImageParametersAndReferenceCountsFailBeforePayment() {
        var model = model("seedream-5-5");
        var task = ImageTask.builder().prompt("蓝色立方体").width(1024).height(1024).build();
        assertThat(ApixoGenerationRequests.image(task, model).getStr("size")).isEqualTo("1024x1024");
        task.setWidth(256);
        assertThatThrownBy(() -> ApixoGenerationRequests.image(task, model)).hasMessageContaining("图片尺寸");
        task.setWidth(null);
        task.setHeight(null);
        task.setResolution("4k");
        assertThatThrownBy(() -> ApixoGenerationRequests.image(task, model)).hasMessageContaining("分辨率");
        task.setResolution("1k");
        task.setRefImageUrls(JSONUtil.toJsonStr(Collections.nCopies(11, "https://example.com/image.png")));
        assertThatThrownBy(() -> ApixoGenerationRequests.image(task, model)).hasMessageContaining("参考图片");
        assertThat(calls).isEmpty();
    }

    @Test
    void failedUnknownMismatchedAndMalformedTasksAreNotReportedAsSuccess() {
        var strategy = new ApixoVideoStrategy();
        var model = model("minimax-h3");
        var task = VideoTask.builder().build();
        state = "failed";
        assertThat(strategy.query("apixo-task-1", task, model, api).state()).isEqualTo(RemoteGenerationResult.State.FAILED);
        state = "unrecognized";
        assertThatThrownBy(() -> strategy.query("apixo-task-1", task, model, api)).hasMessageContaining("尚未确认");
        state = "success";
        returnedId = "different-task";
        assertThatThrownBy(() -> strategy.query("apixo-task-1", task, model, api)).hasMessageContaining("编号不一致");
        returnedId = "apixo-task-1";
        result = "{\"resultUrls\":[\"file:///private/result.mp4\"]}";
        assertThatThrownBy(() -> strategy.query("apixo-task-1", task, model, api)).hasMessageContaining("地址无效");
        result = "{\"resultUrls\":[]}";
        assertThatThrownBy(() -> strategy.query("apixo-task-1", task, model, api)).hasMessageContaining("未返回");
    }

    @Test
    void rejectedAndDisconnectedSubmissionsDoNotCreateAnotherPaidRequest() {
        code = 402;
        var strategy = new ApixoImageStrategy();
        var task = ImageTask.builder().prompt("蓝色立方体").build();
        try (var prepared = strategy.prepare(task, model("seedream-5-5"), api)) {
            var rejected = prepared.submit();
            assertThat(rejected.providerTaskId()).isNull();
            assertThat(rejected.result().state()).isEqualTo(RemoteGenerationResult.State.FAILED);
        }
        assertThat(calls).hasSize(1);
        disconnect = true;
        try (var prepared = strategy.prepare(task, model("seedream-5-5"), api)) {
            assertThatThrownBy(prepared::submit).isInstanceOf(RuntimeException.class);
        }
        assertThat(calls).hasSize(2);
    }

    @Test
    void zeroBalanceStillAllowsReadOnlyModelConfigurationButNoGenerationIsSent() throws Exception {
        var catalog = new ProviderCatalogService(new ObjectMapper());
        var models = new ApixoModelDiscovery(PRESETS).discover(api, catalog.require("apixo"));
        assertThat(models.models()).hasSize(43);
        assertThat(models.models()).filteredOn(model -> model.modelType() == 1).hasSize(25);
        assertThat(models.models()).filteredOn(model -> model.id().equals("gpt-5.6-sol"))
                .singleElement().satisfies(model -> assertThat(model.modelProtocol()).isEqualTo("openai_responses"));
        assertThat(calls).singleElement().satisfies(call -> {
            assertThat(call.method()).isEqualTo("GET");
            assertThat(call.path()).isEqualTo("/api/v1/apikeys/current-balance");
        });
        assertThat(catalog.list().subList(0, 4)).extracting(provider -> provider.id())
                .containsExactly("infistar", "metaso", "erchun", "apixo");
        assertThatThrownBy(() -> catalog.validateProtocol(ApiConfig.builder().providerId("erchun").build(), "apixo"))
                .hasMessageContaining("仅适用于");
    }

    @Test
    void networkCallsRejectAnOuterDatabaseTransaction() {
        TransactionSynchronizationManager.setActualTransactionActive(true);
        var strategy = new ApixoImageStrategy();
        var task = ImageTask.builder().prompt("蓝色立方体").build();
        try (var prepared = strategy.prepare(task, model("seedream-5-5"), api)) {
            assertThatThrownBy(prepared::submit).isInstanceOf(IllegalTransactionStateException.class);
        }
        assertThat(calls).isEmpty();
    }

    @Test
    void invalidDiscoveryKeyExplainsHowToContinueWithoutExposingTheRawResponse() throws Exception {
        code = 401;
        var catalog = new ProviderCatalogService(new ObjectMapper());
        assertThatThrownBy(() -> new ApixoModelDiscovery(PRESETS).discover(api, catalog.require("apixo")))
                .hasMessage("APIXO 密钥无效，请检查后重试");
    }

    private static AiModel model(String name) {
        var preset = PRESETS.getPreset("apixo:" + name);
        return AiModel.builder().code(preset.getStr("modelCode")).modelType(preset.getInt("modelType"))
                .modelProtocol("apixo").capabilityPresetCode(preset.getStr("code"))
                .config(preset.getJSONObject("config").toString()).build();
    }
}
