package com.stonewu.fusion.service.generation.video.strategy.newapi;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicReference;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class NewApiVideoStrategyTests {
    @Test void passesFrozenCapabilityConfigToProtocolAndReturnsRemoteReceiptWithoutWritingItems() throws Exception {
        var server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/v1/video/generations", exchange -> {
            byte[] body = "{\"id\":\"platform-task\",\"task_id\":\"platform-task\"}".getBytes(StandardCharsets.UTF_8);
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, body.length);
            try (var output = exchange.getResponseBody()) { output.write(body); }
        });
        server.start();
        try {
            var models = mock(AiModelService.class);
            var apis = mock(ApiConfigService.class);
            var records = mock(VideoGenerationService.class);
            var metadata = mock(AiModelMetadataResolver.class);
            var capabilities = mock(GenerationModelCapabilityService.class);
            var protocols = mock(NewApiVideoProtocolRouter.class);
            var adapter = mock(NewApiVideoProtocolAdapter.class);
            var model = AiModel.builder().id(10L).code("kling-v3").modelProtocol("kling").apiConfigId(20L).build();
            var api = ApiConfig.builder().id(20L).platform("newapi").apiUrl("http://localhost:" + server.getAddress().getPort()).apiKey("test-key").build();
            var task = VideoTask.builder().id(30L).taskId("video-task").modelId(10L).prompt("cinematic city").build();
            var frozen = new JSONObject().set("forwardGenerationFlagsViaMetadata", true).set("frameRate", 30);
            var captured = new AtomicReference<NewApiVideoProtocolContext>();
            when(capabilities.getMergedModelConfig(model)).thenReturn(frozen);
            when(metadata.resolve(model, api)).thenReturn(new AiModelMetadata("newapi", "newapi", "kling"));
            when(protocols.resolve(any())).thenAnswer(call -> { captured.set(call.getArgument(0)); return adapter; });
            when(adapter.resolveSubmitPath(any())).thenReturn("/v1/video/generations");
            when(adapter.buildSubmitBody(any())).thenReturn(new JSONObject().set("prompt", "cinematic city"));
            var strategy = new NewApiVideoStrategy(models, apis, records, metadata, capabilities, protocols);
            var receipt = strategy.prepare(task, model, api).submit();
            assertThat(receipt.providerTaskId()).isEqualTo("platform-task");
            assertThat((Object) captured.get().modelConfig()).isSameAs(frozen);
            verifyNoInteractions(records, models, apis);
        } finally { server.stop(0); }
    }
}
