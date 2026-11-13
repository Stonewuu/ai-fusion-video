package com.stonewu.fusion.service.ai.youkou;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.catalog.ProviderCatalogService;
import com.stonewu.fusion.service.ai.catalog.YoukouModelDiscovery;
import com.stonewu.fusion.service.generation.image.strategy.youkou.YoukouImageStrategy;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.video.strategy.youkou.YoukouVideoStrategy;
import com.stonewu.fusion.service.media.MediaProbe;
import com.stonewu.fusion.service.media.ProviderMediaFetcher;
import com.stonewu.fusion.service.media.model.MediaFacts;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.transaction.support.TransactionSynchronizationManager;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** 保护平台异步合同、素材编号、规格选择和只提交一次的可观察 HTTP 行为。 */
class YoukouProtocolTests {
    private static final String IMAGE = "GK Image 2.0-BR-default";
    private static final String VIDEO = "2.0 渠道特价";
    private static final String MEDIA = "med_0123456789abcdef0123456789abcdef";
    private final CopyOnWriteArrayList<Call> calls = new CopyOnWriteArrayList<>();
    private ObjectNode catalog;
    private ApiConfig api;
    private AiModel model;
    private HttpServer server;
    private ProviderMediaFetcher fetcher;
    private MediaProbe probe;
    private YoukouGenerationRequests requests;
    private String state = "queued";
    private String wrongIdentity;
    private int createStatus = 202;
    private int queryStatus = 200;
    private boolean disconnect;
    private boolean base64;

    @BeforeEach
    void start() throws Exception {
        try (var input = getClass().getResourceAsStream("/youkou/catalog.json")) {
            catalog = (ObjectNode) YoukouContract.JSON.readTree(input);
        }
        fetcher = mock(ProviderMediaFetcher.class);
        probe = mock(MediaProbe.class);
        requests = new YoukouGenerationRequests(new YoukouReferenceUploads(fetcher, probe));
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            String path = exchange.getRequestURI().getPath();
            var request = exchange.getRequestHeaders();
            calls.add(new Call(exchange.getRequestMethod(), path, request.getFirst("Authorization"),
                    request.getFirst("Idempotency-Key"), new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)));
            if (disconnect && exchange.getRequestMethod().equals("POST")) {
                exchange.close();
                return;
            }
            int status = 200;
            ObjectNode body;
            if (path.equals("/v1/models")) {
                body = YoukouContract.JSON.createObjectNode();
                var data = body.putArray("data");
                catalog.path("data").forEach(row -> data.addObject().put("id", row.path("id").asText()));
            } else if (path.equals("/v1/catalog")) {
                body = catalog;
            } else if (path.equals("/v1/pricing")) {
                body = YoukouContract.JSON.createObjectNode().put("catalog_version", "youkou-test-catalog");
                var data = body.putArray("data");
                catalog.path("data").forEach(row -> data.addObject().put("public_model_id", row.path("public_model_id").asText()));
            } else if (path.equals("/v1/media/uploads")) {
                status = 201;
                body = YoukouContract.JSON.createObjectNode().put("media_id", MEDIA);
            } else if (path.endsWith("/result")) {
                body = YoukouContract.JSON.createObjectNode();
                var item = body.putArray("data").addObject();
                if (base64) {
                    item.put("b64_json", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7lcAAAAASUVORK5CYII=");
                } else {
                    item.put("url", "https://cdn.example.com/generated.png");
                }
            } else {
                status = exchange.getRequestMethod().equals("POST") ? createStatus : queryStatus;
                body = receipt();
                if (status == 429) {
                    exchange.getResponseHeaders().set("Retry-After", "137");
                }
            }
            byte[] bytes = body.toString().getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "application/json");
            exchange.sendResponseHeaders(status, bytes.length);
            exchange.getResponseBody().write(bytes);
            exchange.close();
        });
        server.start();
        api = ApiConfig.builder().providerId("erchun").platform("openai_compatible")
                .apiUrl("http://127.0.0.1:" + server.getAddress().getPort())
                .apiKey("test-private-key").imageProtocol("youkou").videoProtocol("youkou").build();
        model = model(IMAGE, 2);
    }

    @AfterEach
    void stop() {
        TransactionSynchronizationManager.setActualTransactionActive(false);
        server.stop(0);
    }

    @Test
    void imageSavesIdempotencyBeforePostingThenReadsTheSeparateResult() {
        var strategy = new YoukouImageStrategy(requests);
        var task = ImageTask.builder().prompt("白色背景上的蓝色立方体").resolution("1k").ratio("1:1").build();
        try (var prepared = strategy.prepare(task, model, api)) {
            assertThat(calls).isEmpty();
            assertThat(prepared.intent().payload().toString()).doesNotContain("test-private-key", "Authorization");
            assertThat(prepared.submit().providerTaskId()).isEqualTo("123");
            assertThat(calls.getFirst().idempotency()).isEqualTo(prepared.intent().payload().path("idempotencyKey").asText());
            ObjectNode submitted = YoukouContract.object(calls.getFirst().body(), "测试请求无效");
            assertThat(submitted.path("model").asText()).isEqualTo(IMAGE);
            assertThat(submitted.path("resolution").asText()).isEqualTo("1k");
            assertThat(submitted.path("aspect_ratio").asText()).isEqualTo("1:1");
            assertThat(submitted.path("n").asInt()).isEqualTo(1);
            assertThat(submitted.has("size")).isFalse();
            assertThatThrownBy(prepared::submit).hasMessageContaining("不能再次调用");
        }
        assertThat(strategy.query("123", task, model, api).state()).isEqualTo(RemoteGenerationResult.State.PENDING);
        state = "completed";
        assertThat(strategy.query("123", task, model, api).outputs().getFirst().media())
                .isEqualTo("https://cdn.example.com/generated.png");
        assertThat(calls).extracting(Call::path).containsExactly("/v1/images/generations", "/v1/tasks/123",
                "/v1/tasks/123", "/v1/tasks/123/result");
        assertThat(calls).extracting(Call::authorization).containsOnly("Bearer test-private-key");
    }

    @Test
    void imageBase64ResultIsPassedToManagedMediaWithItsFileType() {
        base64 = true;
        state = "completed";
        var result = new YoukouImageStrategy(requests).query("123", ImageTask.builder().build(), model, api);
        assertThat(result.outputs().getFirst().media()).startsWith("data:image/png;base64,iVBOR");
    }

    @Test
    void videoUploadsReferencesBeforeCreatingAndReturnsTheAuthenticatedContentEndpoint() throws Exception {
        model = model(VIDEO, 3);
        doAnswer(invocation -> {
            Files.write(invocation.getArgument(2, Path.class), new byte[]{1, 2, 3});
            return null;
        }).when(fetcher).copyPublic(anyString(), any(ApiConfig.class), any(Path.class));
        when(probe.inspect(any(Path.class), anyString())).thenReturn(new MediaFacts(
                MediaKind.IMAGE, "image/png", 3, "test-hash", 512, 512, null, false));
        var task = VideoTask.builder().prompt("固定镜头，蓝色立方体缓慢旋转").duration(4).resolution("480P")
                .ratio("16:9").referenceImageUrls("[\"data:image/png;base64,AQID\"]").build();
        var strategy = new YoukouVideoStrategy(requests);
        try (var prepared = strategy.prepare(task, model, api)) {
            assertThat(calls).extracting(Call::path).containsExactly("/v1/media/uploads");
            assertThat(calls.getFirst().body()).contains("name=\"file\"").doesNotContain("name=\"purpose\"");
            assertThat(prepared.intent().payload().path("body").path("input_images").get(0).path("media_id").asText())
                    .isEqualTo(MEDIA);
            prepared.submit();
        }
        state = "completed";
        var result = strategy.query("123", task, model, api);
        assertThat(result.outputs().getFirst().media()).isEqualTo(api.getApiUrl() + "/v1/tasks/123/content");
        assertThat(result.outputs().getFirst().cover()).isNull();
        assertThat(calls).extracting(Call::path).containsExactly("/v1/media/uploads", "/v1/videos", "/v1/tasks/123");
    }

    @Test
    void invalidSelectionAndUnknownFieldsNeverUploadOrCreate() {
        var strategy = new YoukouImageStrategy(requests);
        assertThatThrownBy(() -> strategy.prepare(ImageTask.builder().prompt("立方体").resolution("8k")
                .ratio("1:1").build(), model, api)).hasMessageContaining("分辨率");
        var config = YoukouContract.object(model.getConfig(), "测试配置无效");
        config.putObject("youkouOptions").put("unpublished_option", true);
        model.setConfig(config.toString());
        assertThatThrownBy(() -> strategy.prepare(ImageTask.builder().prompt("立方体").build(), model, api))
                .hasMessageContaining("尚未开放");
        assertThat(calls).isEmpty();
    }

    @Test
    void imageEditingUsesUploadedReferencesAndTheEditsEndpoint() throws Exception {
        doAnswer(invocation -> {
            Files.write(invocation.getArgument(2, Path.class), new byte[]{1, 2, 3});
            return null;
        }).when(fetcher).copyPublic(anyString(), any(ApiConfig.class), any(Path.class));
        when(probe.inspect(any(Path.class), anyString())).thenReturn(new MediaFacts(
                MediaKind.IMAGE, "image/png", 3, "test-image", 512, 512, null, false));
        var task = ImageTask.builder().prompt("保留构图，将立方体改成绿色").resolution("1k").ratio("1:1")
                .refImageUrls("[\"data:image/png;base64,AQID\"]").build();
        try (var prepared = new YoukouImageStrategy(requests).prepare(task, model, api)) {
            prepared.submit();
        }
        assertThat(calls).extracting(Call::path).containsExactly("/v1/media/uploads", "/v1/images/edits");
        var body = YoukouContract.object(calls.getLast().body(), "测试请求无效");
        assertThat(body.path("input_images").get(0).path("media_id").asText()).isEqualTo(MEDIA);
        assertThat(body.has("image")).isFalse();
        assertThat(body.toString()).doesNotContain("base64", "data:image");
    }

    @Test
    void mixedReferencesAreUploadedSeparatelyAndSubmittedInTheirOwnFields() throws Exception {
        model = model(VIDEO, 3);
        doAnswer(invocation -> {
            Files.writeString(invocation.getArgument(2, Path.class), invocation.getArgument(0, String.class));
            return null;
        }).when(fetcher).copyPublic(anyString(), any(ApiConfig.class), any(Path.class));
        when(probe.inspect(any(Path.class), anyString())).thenAnswer(invocation -> {
            String source = Files.readString(invocation.getArgument(0, Path.class));
            if (source.startsWith("data:audio/")) {
                return new MediaFacts(MediaKind.AUDIO, "audio/mpeg", 100, "audio-hash", null, null, 1500L, true);
            }
            if (source.startsWith("data:video/")) {
                return new MediaFacts(MediaKind.VIDEO, "video/mp4", 100, "video-hash", 512, 512, 2000L, false);
            }
            return new MediaFacts(MediaKind.IMAGE, "image/png", 100, "image-hash", 512, 512, null, false);
        });
        var task = VideoTask.builder().prompt("参照画面和节奏生成视频").duration(4).resolution("480P").ratio("16:9")
                .referenceImageUrls("[\"data:image/png;base64,AQID\"]")
                .referenceVideoUrls("[\"data:video/mp4;base64,AQID\"]")
                .referenceAudioUrls("[\"data:audio/mpeg;base64,AQID\"]").build();
        try (var prepared = new YoukouVideoStrategy(requests).prepare(task, model, api)) {
            assertThat(calls).hasSize(3).allMatch(call -> call.path().equals("/v1/media/uploads"));
            prepared.submit();
        }
        var body = YoukouContract.object(calls.getLast().body(), "测试请求无效");
        for (String field : List.of("input_images", "input_videos", "input_audios")) {
            assertThat(body.path(field).size()).isEqualTo(1);
            assertThat(body.path(field).get(0).fieldNames()).toIterable().containsExactly("media_id");
        }
        assertThat(body.toString()).doesNotContain("data:", "input_reference", "input_image_files");
        assertThat(calls.getLast().path()).isEqualTo("/v1/videos");
    }

    @Test
    void createNetworkFailureAndWrongSuccessStatusAreNeverRetried() {
        disconnect = true;
        var strategy = new YoukouImageStrategy(requests);
        try (var prepared = strategy.prepare(ImageTask.builder().prompt("立方体").build(), model, api)) {
            assertThatThrownBy(prepared::submit).hasMessageContaining("连接中断");
            assertThatThrownBy(prepared::submit).hasMessageContaining("不能再次调用");
        }
        assertThat(calls).hasSize(1);
        disconnect = false;
        createStatus = 200;
        try (var prepared = strategy.prepare(ImageTask.builder().prompt("立方体").build(), model, api)) {
            assertThatThrownBy(prepared::submit).hasMessageContaining("非预期");
        }
        assertThat(calls).hasSize(2);
    }

    @Test
    void queryDistinguishesRateLimitsFailureAndMismatchedReceipts() {
        var strategy = new YoukouImageStrategy(requests);
        var task = ImageTask.builder().build();
        queryStatus = 429;
        assertThat(strategy.query("123", task, model, api).nextPollSeconds()).isEqualTo(137);
        queryStatus = 200;
        state = "failed";
        assertThat(strategy.query("123", task, model, api).error()).isEqualTo("模型暂不可用");
        wrongIdentity = "mdl_00000000000000000000000000000000";
        assertThatThrownBy(() -> strategy.query("123", task, model, api)).hasMessageContaining("不一致");
        assertThat(calls).allMatch(call -> call.method().equals("GET"));
    }

    @Test
    void discoveryUsesPublishedModelIdentityAndCarriesTextCapabilities() throws Exception {
        var provider = new ProviderCatalogService(YoukouContract.JSON).require("erchun");
        var result = new YoukouModelDiscovery().discover(api, provider);
        assertThat(result.models()).hasSize(3);
        assertThat(result.models()).allMatch(candidate -> candidate.configurationHint() == null);
        assertThat(result.models()).allMatch(candidate -> !YoukouContract.object(candidate.config(), "模型配置无法读取").has("maxCount"));
        var text = result.models().stream().filter(candidate -> candidate.modelType() == 1).findFirst().orElseThrow();
        assertThat(text.textCapabilities().contextWindow()).isEqualTo(1_000_000);
        assertThat(text.textCapabilities().reasoningEfforts()).containsExactly("none", "low", "high", "max");
        assertThat(text.textCapabilities().inputTypes()).isEmpty();
        assertThat(text.config()).contains("chat_completions").doesNotContain("group_prices");
        assertThat(calls).extracting(Call::path).containsExactly("/v1/models", "/v1/catalog", "/v1/pricing");
    }

    @Test
    void networkPreparationRejectsAnOuterDatabaseTransaction() {
        TransactionSynchronizationManager.setActualTransactionActive(true);
        assertThatThrownBy(() -> requests.image(ImageTask.builder().prompt("立方体").build(), model, api))
                .hasMessageContaining("不能持有数据库事务");
        assertThat(calls).isEmpty();
    }

    private ObjectNode receipt() {
        var contract = YoukouContract.from(model);
        String identity = wrongIdentity == null ? contract.identity() : wrongIdentity;
        ObjectNode result = YoukouContract.JSON.createObjectNode().put("id", "123").put("object", "generation.task")
                .put("model", identity).put("public_model_id", identity).put("public_model_name", model.getCode())
                .put("output_type", contract.kind()).put("status", state).put("status_url", "/v1/tasks/123")
                .put("result_url", "/v1/tasks/123/result").put("content_url", "/v1/tasks/123/content");
        if (state.equals("failed")) {
            result.putObject("error").put("code", "model_unavailable").put("message", "模型暂不可用");
        }
        return result;
    }

    private AiModel model(String id, int type) {
        JsonNode entry = catalog.path("data").valueStream().filter(row -> row.path("id").asText().equals(id))
                .findFirst().orElseThrow();
        String endpoint = type == 2 ? "/v1/images/generations" : "/v1/videos";
        return AiModel.builder().code(id).modelType(type).modelProtocol("youkou")
                .config(YoukouModelCapabilities.configuration(entry, endpoint).toString()).build();
    }

    private record Call(String method, String path, String authorization, String idempotency, String body) {
    }
}
