package com.stonewu.fusion.service.generation.video.strategy.infistar;

import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.generation.VideoModelRules;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.media.MediaRemoteAddressPolicy;
import com.stonewu.fusion.service.media.ProviderMediaFetcher;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolContext;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import okio.Buffer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import com.sun.net.httpserver.HttpServer;

import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;

class InfistarVideoProtocolTests {
    /** 手动放开能力开关也不能让未编码的素材在提交时被静默丢弃。 */
    @ParameterizedTest
    @ValueSource(strings = {"lastFrame", "images", "videos", "audios"})
    void unsupportedMediaIsRejectedBeforeSubmittingInsteadOfDropped(String input) {
        var task = VideoTask.builder().prompt("根据素材生成视频").duration(5).build();
        switch (input) {
            case "lastFrame" -> task.setLastFrameImageUrl("https://example.com/last.png");
            case "images" -> task.setReferenceImageUrls("[\"https://example.com/reference.png\"]");
            case "videos" -> task.setReferenceVideoUrls("[\"https://example.com/reference.mp4\"]");
            case "audios" -> task.setReferenceAudioUrls("[\"https://example.com/reference.mp3\"]");
            default -> throw new IllegalArgumentException(input);
        }
        var config = JSONUtil.parseObj("""
                {"supportLastFrame":true,"supportReferenceImages":true,
                 "supportReferenceVideos":true,"supportReferenceAudios":true}
                """);
        var context = new OpenAiCompatibleVideoProtocolContext(AiModel.builder().code("doubao-seedance-2-0-260128").build(),
                ApiConfig.builder().apiUrl("https://example.com").build(), task, config, null);
        var adapter = new InfistarVideoProtocolAdapter(new OpenAiCompatibleVideoProtocolSupport(
                mock(StorageConfigService.class), mock(PresetArtStyleResourceResolver.class)));
        assertThatThrownBy(() -> adapter.buildSubmitBody(context)).hasMessageContaining("不能忽略后继续提交");
    }

    @TempDir
    Path temporary;

    /** 未填写时长的万相请求实际发送五秒；提交、轮询、鉴权下载保持同一条 HTTP 合同。 */
    @Test
    void runsTheAsyncContractAndDownloadsTheAuthenticatedContent() throws Exception {
        var seen = new CopyOnWriteArrayList<String>();
        var submitted = new CopyOnWriteArrayList<String>();
        var queried = new AtomicInteger();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/videos", exchange -> {
            seen.add(exchange.getRequestMethod() + " " + exchange.getRequestURI().getPath() + " "
                    + exchange.getRequestHeaders().getFirst("Authorization"));
            String requestBody = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            if (exchange.getRequestMethod().equals("POST")) {
                submitted.add(requestBody);
            }
            boolean download = exchange.getRequestURI().getPath().endsWith("/content");
            String response = download ? "fixture-video-bytes" : exchange.getRequestMethod().equals("POST")
                    ? "{\"task_id\":\"video-fixture\",\"status\":\"queued\"}"
                    : "{\"id\":\"video-fixture\",\"status\":\""
                    + (queried.incrementAndGet() == 1 ? "in_progress" : "completed") + "\",\"seconds\":\"5\"}";
            byte[] body = response.getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", download ? "video/mp4" : "application/json");
            exchange.sendResponseHeaders(200, body.length);
            exchange.getResponseBody().write(body);
            exchange.close();
        });
        server.start();
        try {
            var presets = new ModelPresetService();
            presets.init();
            var model = AiModel.builder().code("wan3.0-video").modelType(3).modelProtocol("infistar_video")
                    .capabilityPresetCode("infistar:wan3.0-video").build();
            var task = VideoTask.builder().prompt("雨后的街道").build();
            VideoModelRules.normalizeTask(JSONUtil.parseObj(presets.getPresetConfig(model.getCapabilityPresetCode())), task);
            var api = ApiConfig.builder().apiUrl("http://127.0.0.1:" + server.getAddress().getPort())
                    .apiKey("fixture-key").platform("openai_compatible").build();
            var configs = mock(ApiConfigService.class);
            var support = new OpenAiCompatibleVideoProtocolSupport(mock(StorageConfigService.class),
                    mock(PresetArtStyleResourceResolver.class));
            var strategy = new InfistarVideoStrategy(mock(AiModelService.class), configs,
                    mock(VideoGenerationService.class), presets, new AiModelMetadataResolver(configs),
                    support, new InfistarVideoProtocolAdapter(support));
            try (var prepared = strategy.prepare(task, model, api)) {
                assertThat(prepared.submit().providerTaskId()).isEqualTo("video-fixture");
            }
            var pending = strategy.query("video-fixture", task, model, api);
            assertThat(pending.state()).isEqualTo(RemoteGenerationResult.State.PENDING);
            assertThat(pending.outputs()).isEmpty();
            var finished = strategy.query("video-fixture", task, model, api);
            assertThat(finished.state()).isEqualTo(RemoteGenerationResult.State.SUCCEEDED);
            Path output = temporary.resolve("result.mp4");
            new ProviderMediaFetcher(mock(MediaRemoteAddressPolicy.class))
                    .copy(finished.outputs().getFirst().media(), api, output);
            assertThat(Files.readString(output)).isEqualTo("fixture-video-bytes");
            assertThat(seen).containsExactly("POST /v1/videos Bearer fixture-key",
                    "GET /v1/videos/video-fixture Bearer fixture-key", "GET /v1/videos/video-fixture Bearer fixture-key",
                    "GET /v1/videos/video-fixture/content Bearer fixture-key");
            assertThat(submitted).singleElement().asString()
                    .containsPattern("name=\"model\"\r\n(?:[^\r\n]+\r\n)*\r\nwan3.0-video\r\n")
                    .containsPattern("name=\"seconds\"\r\n(?:[^\r\n]+\r\n)*\r\n5\r\n");
        } finally {
            server.stop(0);
        }
    }

    /** 参数面板选择的档位和比例必须实际转换为像素尺寸，不能被请求构造器忽略。 */
    @Test
    void multipartUsesDocumentedFieldsAndDoesNotInventNativeCapabilities() throws Exception {
        var presets = new ModelPresetService();
        presets.init();
        var config = JSONUtil.parseObj(presets.getPresetConfig("infistar:MiniMax-H3"));
        var task = VideoTask.builder().prompt("海边的灯塔").duration(5).build();
        VideoModelRules.normalizeTask(config, task);
        var api = ApiConfig.builder().apiUrl("https://example.com/v1").apiKey("fixture-key").build();
        var context = new OpenAiCompatibleVideoProtocolContext(AiModel.builder().code("MiniMax-H3").build(),
                api, task, config, null);
        var adapter = new InfistarVideoProtocolAdapter(new OpenAiCompatibleVideoProtocolSupport(
                mock(StorageConfigService.class), mock(PresetArtStyleResourceResolver.class)));
        var buffer = new Buffer();
        adapter.buildSubmitBody(context).writeTo(buffer);
        assertThat(buffer.readUtf8()).contains("name=\"model\"", "MiniMax-H3", "name=\"seconds\"", "海边的灯塔")
                .contains("name=\"size\"", "1366x768")
                .doesNotContain("平台默认", "aspect_ratio", "fast_mode");
        assertThat(adapter.resolveSubmitUrl(context)).isEqualTo("https://example.com/v1/videos");
        var result = adapter.parseSubmitResponse(context, "{\"task_id\":\"task-1\",\"status\":\"queued\"}");
        assertThat(adapter.resolveVideoContentUrl(context, "task-1", result))
                .isEqualTo("https://example.com/v1/videos/task-1/content");
        assertThatThrownBy(() -> adapter.parseQueryResponse(context,
                "{\"task_id\":\"task-1\",\"id\":\"another\"}")) .hasMessageContaining("冲突");
        task.setRatio("9:16");
        var portrait = new Buffer();
        adapter.buildSubmitBody(context).writeTo(portrait);
        assertThat(portrait.readUtf8()).contains("768x1366");
        task.setRatio("7:1");
        assertThatThrownBy(() -> adapter.buildSubmitBody(context)).hasMessageContaining("视频尺寸");
    }
}
