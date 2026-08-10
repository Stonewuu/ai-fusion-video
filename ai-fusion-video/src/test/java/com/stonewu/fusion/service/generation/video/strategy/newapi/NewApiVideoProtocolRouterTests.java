package com.stonewu.fusion.service.generation.video.strategy.newapi;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class NewApiVideoProtocolRouterTests {

    @Test
    void shouldResolveDedicatedProtocolAdapterWhenPresent() {
        NewApiVideoProtocolAdapter genericAdapter = mock(NewApiVideoProtocolAdapter.class);
        when(genericAdapter.getProtocol()).thenReturn("newapi");
        NewApiVideoProtocolAdapter jimengAdapter = mock(NewApiVideoProtocolAdapter.class);
        when(jimengAdapter.getProtocol()).thenReturn("jimeng");

        NewApiVideoProtocolRouter router = new NewApiVideoProtocolRouter(List.of(genericAdapter, jimengAdapter));
        NewApiVideoProtocolContext context = new NewApiVideoProtocolContext(
                AiModel.builder().code("jimeng-v1").build(),
                null,
                null,
                new JSONObject(),
                new AiModelMetadata("newapi", "newapi", "jimeng")
        );

        assertSame(jimengAdapter, router.resolve(context));
    }

    @Test
    void shouldRejectMissingDedicatedProtocolInsteadOfFallingBack() {
        NewApiVideoProtocolAdapter genericAdapter = mock(NewApiVideoProtocolAdapter.class);
        when(genericAdapter.getProtocol()).thenReturn("newapi");

        NewApiVideoProtocolRouter router = new NewApiVideoProtocolRouter(List.of(genericAdapter));
        NewApiVideoProtocolContext context = new NewApiVideoProtocolContext(
                AiModel.builder().code("kling-v1").build(),
                null,
                null,
                new JSONObject(),
                new AiModelMetadata("newapi", "newapi", "kling")
        );

        assertThatThrownBy(() -> router.resolve(context))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("kling");
    }

    @Test
    void shouldUseDedicatedKlingPathsAndForwardNamedResolution() {
        NewApiVideoProtocolSupport support = new NewApiVideoProtocolSupport();
        NewApiKlingVideoProtocolAdapter adapter = new NewApiKlingVideoProtocolAdapter(support);
        JSONObject modelConfig = new JSONObject()
                .set("frameRate", 30)
                .set("forwardGenerationFlagsViaMetadata", true);
        VideoTask task = VideoTask.builder()
                .prompt("cinematic city")
                .firstFrameImageUrl("https://example.com/first.png")
                .duration(5)
                .resolution("1080p")
                .ratio("9:16")
                .generateAudio(true)
                .build();
        NewApiVideoProtocolContext context = new NewApiVideoProtocolContext(
                AiModel.builder().code("kling-v3").build(),
                null,
                task,
                modelConfig,
                new AiModelMetadata("newapi", "newapi", "kling")
        );

        JSONObject body = adapter.buildSubmitBody(context);

        assertEquals("/kling/v1/videos/image2video", adapter.resolveSubmitPath(context));
        assertEquals("/kling/v1/videos/image2video/task-123",
                adapter.resolveQueryPath(context, "task-123"));
        assertEquals(1080, body.getInt("width"));
        assertEquals(1920, body.getInt("height"));
        assertEquals(30, body.getInt("fps"));
        assertTrue(body.getJSONObject("metadata").getBool("generate_audio"));
    }

    @Test
    void shouldUseKlingTextPathWithoutImageInput() {
        NewApiKlingVideoProtocolAdapter adapter =
                new NewApiKlingVideoProtocolAdapter(new NewApiVideoProtocolSupport());
        NewApiVideoProtocolContext context = new NewApiVideoProtocolContext(
                AiModel.builder().code("kling-3.0-turbo").build(),
                null,
                VideoTask.builder().prompt("ocean waves").build(),
                new JSONObject(),
                new AiModelMetadata("newapi", "newapi", "kling")
        );

        assertEquals("/kling/v1/videos/text2video", adapter.resolveSubmitPath(context));
    }
}
