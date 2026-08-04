package com.stonewu.fusion.service.generation.video.strategy.agnes;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolContext;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoTaskResult;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import okio.Buffer;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class AgnesVideoProtocolTests {

    private final OpenAiCompatibleVideoProtocolSupport support =
            new OpenAiCompatibleVideoProtocolSupport(
                    mock(StorageConfigService.class),
                    new PresetArtStyleResourceResolver());
    private final AgnesVideoProtocolAdapter adapter = new AgnesVideoProtocolAdapter(support);

    @Test
    void mapsResolutionTierAndOptionalParametersIntoSubmitBody() throws Exception {
        VideoTask task = VideoTask.builder()
                .prompt("cinematic ocean wave at sunset")
                .duration(5)
                .ratio("16:9")
                .resolution("720p")
                .seed(42L)
                .firstFrameImageUrl("https://example.com/first.png")
                .build();
        JSONObject modelConfig = JSONUtil.parseObj("""
                {
                  "frameRate": 24,
                  "numInferenceSteps": 30,
                  "negativePrompt": "blurry, low quality",
                  "defaultAspectRatio": "16:9"
                }
                """);

        String body = readBody(adapter.buildSubmitBody(context(task, modelConfig)));
        JSONObject json = JSONUtil.parseObj(body);

        assertThat(json.getStr("model")).isEqualTo("agnes-video-v2.0");
        assertThat(json.getStr("prompt")).isEqualTo("cinematic ocean wave at sunset");
        assertThat(json.getInt("width")).isEqualTo(1280);
        assertThat(json.getInt("height")).isEqualTo(720);
        assertThat(json.getInt("frame_rate")).isEqualTo(24);
        assertThat(json.getInt("num_frames")).isEqualTo(121);
        assertThat(json.getInt("num_inference_steps")).isEqualTo(30);
        assertThat(json.getStr("negative_prompt")).isEqualTo("blurry, low quality");
        assertThat(json.getLong("seed")).isEqualTo(42L);
        assertThat(json.getStr("image")).isEqualTo("https://example.com/first.png");
        assertThat(json.getStr("mode")).isEqualTo("ti2vid");
        assertThat(json.containsKey("extra_body")).isFalse();
    }

    @Test
    void keyframeModePutsImagesAndModeUnderExtraBody() throws Exception {
        VideoTask task = VideoTask.builder()
                .prompt("smooth transition between keyframes")
                .duration(5)
                .firstFrameImageUrl("https://example.com/a.png")
                .lastFrameImageUrl("https://example.com/b.png")
                .build();

        String body = readBody(adapter.buildSubmitBody(context(task, JSONUtil.createObj().set("frameRate", 24))));
        JSONObject json = JSONUtil.parseObj(body);

        assertThat(json.containsKey("image")).isFalse();
        assertThat(json.getJSONObject("extra_body").getJSONArray("image").toList(String.class))
                .containsExactly("https://example.com/a.png", "https://example.com/b.png");
        assertThat(json.getJSONObject("extra_body").getStr("mode")).isEqualTo("keyframes");
    }

    @Test
    void parseQueryResponseReadsCompletedVideoUrlFromMetadata() {
        OpenAiCompatibleVideoTaskResult result = adapter.parseQueryResponse(
                context(VideoTask.builder().prompt("x").build(), JSONUtil.createObj()),
                """
                {
                  "id":"task_123",
                  "video_id":"video_abc",
                  "status":"completed",
                  "seconds":"5.0",
                  "metadata":{
                    "url":"https://platform-outputs.agnes-ai.space/videos/result.mp4",
                    "cover_url":"https://platform-outputs.agnes-ai.space/covers/result.jpg"
                  }
                }
                """);

        assertThat(result.trackingId()).isEqualTo("video_abc");
        assertThat(result.status()).isEqualTo("completed");
        assertThat(result.durationSeconds()).isEqualTo(5);
        assertThat(result.videoUrl()).isEqualTo("https://platform-outputs.agnes-ai.space/videos/result.mp4");
        assertThat(result.coverUrl()).isEqualTo("https://platform-outputs.agnes-ai.space/covers/result.jpg");
    }

    private OpenAiCompatibleVideoProtocolContext context(VideoTask task, JSONObject modelConfig) {
        return new OpenAiCompatibleVideoProtocolContext(
                AiModel.builder().code("agnes-video-v2.0").modelProtocol("agnes").build(),
                ApiConfig.builder()
                        .platform("openai_compatible")
                        .apiUrl("https://apihub.agnes-ai.com")
                        .apiKey("test-key")
                        .build(),
                task,
                modelConfig,
                null
        );
    }

    private String readBody(okhttp3.RequestBody body) throws Exception {
        Buffer buffer = new Buffer();
        body.writeTo(buffer);
        return buffer.readUtf8();
    }
}
