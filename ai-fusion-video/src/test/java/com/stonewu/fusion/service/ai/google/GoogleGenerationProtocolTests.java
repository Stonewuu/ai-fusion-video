package com.stonewu.fusion.service.ai.google;

import cn.hutool.json.JSONUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.generation.audio.AudioTaskData;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.generation.video.strategy.google.GoogleVideoProtocol;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Base64;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 保护 Google 各类原生接口的请求格式、异步结束条件和媒体结果，避免套用文本协议。 */
class GoogleGenerationProtocolTests {
    @Test
    void omniPreservesFrameOrderAndCreatesAnAsyncVideoInteraction() {
        var model = AiModel.builder().code("gemini-omni-1.2-flash").modelType(3).build();
        var task = VideoTask.builder().prompt("镜头缓慢推进").duration(5).ratio("9:16").resolution("720P")
                .firstFrameImageUrl("data:image/png;base64,YQ==").lastFrameImageUrl("data:image/jpeg;base64,Yg==")
                .referenceImageUrls("[\"data:image/png;base64,Yw==\"]").seed(42L).build();
        var request = GoogleVideoProtocol.request(task, model, new JSONObject().set("maxDuration", 15), false);
        assertThat(request.getBool("background")).isTrue();
        assertThat(request.getBool("store")).isTrue();
        assertThat(request.getStr("model")).isEqualTo("gemini-omni-1.2-flash");
        assertThat(request.getByPath("input[0].type", String.class)).isEqualTo("user_input");
        assertThat(request.getByPath("input[0].content[1].data", String.class)).isEqualTo("YQ==");
        assertThat(request.getByPath("input[0].content[3].data", String.class)).isEqualTo("Yg==");
        assertThat(request.getByPath("input[0].content[4].data", String.class)).isEqualTo("Yw==");
        assertThat(request.getByPath("response_format[0].aspect_ratio", String.class)).isEqualTo("9:16");
        assertThat(request.getByPath("response_format[0].duration", String.class)).isEqualTo("5s");
        assertThat(request.getByPath("generation_config.seed", Long.class)).isEqualTo(42);
        var developer = GoogleVideoProtocol.request(task, model, new JSONObject().set("maxDuration", 15), true);
        assertThat(developer.getByPath("input[0].content[1].data", String.class)).isEqualTo("YQ==");
        assertThat(developer.getByPath("response_format.duration", String.class)).isEqualTo("5s");
    }

    @Test
    void veoUsesChannelSpecificInlineMediaAndRejectsInvalidCombinations() {
        var model = AiModel.builder().code("veo-3.1-generate-001").modelType(3).build();
        var task = VideoTask.builder().prompt("水面倒影").duration(8).resolution("1080P")
                .firstFrameImageUrl("data:image/png;base64,YQ==").build();
        assertThat(GoogleVideoProtocol.request(task, model, new JSONObject(), false)
                .getByPath("instances[0].image.bytesBase64Encoded", String.class)).isEqualTo("YQ==");
        assertThat(GoogleVideoProtocol.request(task, model, new JSONObject(), true)
                .getByPath("instances[0].image.inlineData.data", String.class)).isEqualTo("YQ==");
        task.setDuration(4);
        assertThatThrownBy(() -> GoogleVideoProtocol.request(task, model, new JSONObject(), false)).hasMessageContaining("8 秒");
        task.setResolution("720p");
        assertThat(GoogleVideoProtocol.request(task, model, new JSONObject(), false)
                .getByPath("parameters.durationSeconds", Integer.class)).isEqualTo(4);
        task.setDuration(5);
        assertThatThrownBy(() -> GoogleVideoProtocol.request(task, model, new JSONObject(), false)).hasMessageContaining("4、6");
    }

    @Test
    void videoPollingWaitsForCompletionAndNeverPublishesEmptyOutput() {
        assertThat(GoogleVideoProtocol.result(JSONUtil.parseObj("""
                {"done":true,"error":{"code":8,"message":"The service is currently experiencing high load"}}
                """), false, false, 4).error()).contains("繁忙");
        assertThat(GoogleVideoProtocol.result(JSONUtil.parseObj("{\"status\":\"in_progress\"}"), true, false, 5).state())
                .isEqualTo(RemoteGenerationResult.State.PENDING);
        var complete = GoogleVideoProtocol.result(JSONUtil.parseObj("""
                {"status":"completed","steps":[
                  {"type":"thought","content":[{"type":"video","data":"ignored","mime_type":"video/mp4"}]},
                  {"type":"model_output","content":[{"type":"video","data":"YQ==","mime_type":"video/mp4"}]}
                ]}
                """), true, false, 5);
        assertThat(complete.outputs()).hasSize(1);
        assertThat(complete.outputs().getFirst().media()).isEqualTo("data:video/mp4;base64,YQ==");
        assertThat(GoogleVideoProtocol.result(JSONUtil.parseObj("{\"done\":false}"), false, true, 8).state())
                .isEqualTo(RemoteGenerationResult.State.PENDING);
        assertThatThrownBy(() -> GoogleVideoProtocol.result(JSONUtil.parseObj("{\"done\":true,\"response\":{}}"),
                false, false, 8)).hasMessageContaining("没有可用输出");
    }

    @Test
    void speechGenerationsUseTheirDocumentedVoiceContracts() {
        var input = input("你好", "Kore");
        var modern = AiModel.builder().modelType(4).code("gemini-3.8-flash-tts").build();
        var vertex = GoogleAudioProtocol.request(modern, input, List.of(), new JSONObject(), false);
        assertThat(vertex.operation()).isEqualTo("generateContent");
        assertThat(vertex.body().getByPath("generationConfig.speechConfig.voiceConfig.voice", String.class)).isEqualTo("Kore");
        var developer = GoogleAudioProtocol.request(modern, input, List.of(), new JSONObject(), true);
        assertThat(developer.operation()).isEqualTo("interactions");
        assertThat(developer.body().getByPath("generation_config.speech_config[0].voice", String.class)).isEqualTo("Kore");
        var old = GoogleAudioProtocol.request(AiModel.builder().modelType(4).code("gemini-2.5-flash-tts").build(),
                input, List.of(), new JSONObject(), false);
        assertThat(old.body().getByPath("generationConfig.speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName", String.class))
                .isEqualTo("Kore");
    }

    @Test
    void vertexMusicUsesUserInputSteps() {
        var model = AiModel.builder().modelType(4).code("lyria-3-clip-preview").build();
        var request = GoogleAudioProtocol.request(model, input("本次请求", null), List.of(), new JSONObject(), false);
        assertThat(request.body().getByPath("input[0].type", String.class)).isEqualTo("user_input");
        assertThat(request.body().getByPath("input[0].content[0].text", String.class)).isEqualTo("本次请求");
        assertThat(request.body().getBool("store")).isFalse();
        assertThat(request.body().getBool("stream")).isFalse();
    }

    @Test
    void speechResultRequiresAudioAndDoesNotPublishThoughts() {
        var output = GoogleAudioProtocol.output(JSONUtil.parseObj("""
                {"candidates":[{"content":{"parts":[
                  {"text":"你好"},{"inlineData":{"data":"YQ==","mimeType":"audio/pcm;rate=24000"}},
                  {"text":"内部思考","thought":true}
                ]}}]}
                """), "generateContent");
        assertThat(output.text()).isEqualTo("你好");
        assertThat(output.audio()).hasSize(1);
        assertThatThrownBy(() -> GoogleAudioProtocol.output(JSONUtil.parseObj("{\"candidates\":[]}"),
                "generateContent")).isInstanceOf(GoogleAudioProtocol.InvalidOutput.class).hasMessageContaining("未返回音频");
        assertThatThrownBy(() -> GoogleAudioProtocol.output(JSONUtil.parseObj("{\"status\":\"completed\",\"model\":\"lyria-3-clip-preview\"}"),
                "interactions")).isInstanceOf(GoogleAudioProtocol.InvalidOutput.class).hasMessageContaining("未返回音频");
    }

    @Test
    void lyriaSupportsTheObservedPredictOutputAndTheDocumentedAudioContentFormat() {
        var observed = GoogleAudioProtocol.output(JSONUtil.parseObj("{\"predictions\":[{\"bytesBase64Encoded\":\"UklGRg==\"}]}"), "predict");
        assertThat(observed.audio()).containsExactly(new GoogleAudioProtocol.Audio("UklGRg==", "audio/wav"));
        var documented = GoogleAudioProtocol.output(JSONUtil.parseObj("{\"predictions\":[{\"audioContent\":\"UklGRg==\",\"mimeType\":\"audio/wav\"}]}"), "predict");
        assertThat(documented.audio()).isEqualTo(observed.audio());
        assertThatThrownBy(() -> GoogleAudioProtocol.output(JSONUtil.parseObj("{\"predictions\":[{}]}"), "predict"))
                .hasMessageContaining("未返回音频");
    }

    @Test
    void pcmConversionPreservesSamplesAndTheDeclaredDuration() {
        byte[] samples = new byte[48000];
        samples[42] = 7;
        var wave = GoogleAudio.wave(samples, "audio/L16;rate=24000");
        var buffer = ByteBuffer.wrap(wave).order(ByteOrder.LITTLE_ENDIAN);
        assertThat(new String(wave, 0, 4)).isEqualTo("RIFF");
        assertThat(buffer.getInt(24)).isEqualTo(24000);
        assertThat(buffer.getInt(40)).isEqualTo(48000);
        assertThat(wave[86]).isEqualTo((byte) 7);
        assertThat(wave).hasSize(48044);
    }

    @Test
    void officialGoogleRoutesKeepAuthenticationModesAndRegionsExplicit() {
        var google = new GoogleNativeApi();
        var key = ApiConfig.builder().platform("vertex_ai_api_key").apiKey("fixture").appId("test-project").apiUrl("global").build();
        assertThat(google.headers(key)).containsOnlyKeys("x-goog-api-key");
        assertThat(google.interactionsUrl(key)).isEqualTo("https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/interactions");
        key.setPlatform("vertex_ai");
        key.setApiUrl("us");
        assertThat(google.modelUrl(key, "gemini-3.8-flash-tts", "generateContent"))
                .startsWith("https://aiplatform.us.rep.googleapis.com/v1/projects/test-project/locations/us/");
    }

    private AudioTaskData.Input input(String text, String voice) {
        return new AudioTaskData.Input(text, List.of(), voice);
    }
}
