package com.stonewu.fusion.service.ai.catalog;

import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.generation.VideoModelRules;
import com.stonewu.fusion.service.generation.image.strategy.openai.OpenAiImageProtocolAdapter;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolContext;
import com.stonewu.fusion.service.generation.image.strategy.support.OpenAiCompatibleImageProtocolSupport;
import com.stonewu.fusion.service.generation.video.strategy.infistar.InfistarVideoProtocolAdapter;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolContext;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import com.stonewu.fusion.service.storage.StorageConfigService;
import com.stonewu.fusion.service.system.PresetArtStyleResourceResolver;
import okio.Buffer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;

/** 保护默认配置到真实请求参数的契约，不调用付费线路。 */
class InfistarModelDefaultsTests {
    private static final ModelPresetService PRESETS = new ModelPresetService();
    private final ApiConfig api = ApiConfig.builder().apiUrl("https://example.com").apiKey("fixture-key").build();

    @BeforeAll
    static void loadPresets() {
        PRESETS.init();
    }

    @ParameterizedTest
    @CsvSource({
            "gpt-image-2.5-sunburst,1024x1024", "gpt-image-2.5-flare,1024x1024",
            "qwen-image-3.0-pro,2048x2048", "qwen-image-3.0,2048x2048",
            "qwen-image-edit-max,1024x1024", "qwen-image-edit-plus,1024x1024",
            "doubao-seedream-5-0-pro,2048x2048", "doubao-seedream-5-0-flash,2048x2048",
            "doubao-seedream-4-5,2048x2048", "step-2x-large,1024x1024", "step-image-edit-2,1024x1024",
            "grok-imagine-image,1024x1024", "grok-imagine-image-quality,1024x1024",
            "jimeng-image-4.6,2048x2048", "jimeng-image-4.0,2048x2048"
    })
    void addedImagesHaveUsableDefaultGenerationRequests(String code, String expectedSize) throws Exception {
        String presetCode = PRESETS.findProviderPresetCode("infistar", code, 2);
        assertThat(presetCode).isNotBlank();
        var config = JSONUtil.parseObj(PRESETS.getPresetConfig(presetCode));
        var support = new OpenAiCompatibleImageProtocolSupport(PRESETS, mock(StorageConfigService.class),
                new PresetArtStyleResourceResolver());
        var task = ImageTask.builder().prompt("海边的灯塔").ratio("1:1")
                .resolution(config.getStr("defaultResolution")).build();
        int[] size = support.resolveConfiguredSize(task, config);
        var context = new OpenAiCompatibleImageProtocolContext(AiModel.builder().code(code).build(), api,
                code, task.getPrompt(), size[0], size[1], 1, List.of(), config, false);
        var request = new OpenAiImageProtocolAdapter(support).buildRequest(context);
        var buffer = new Buffer();
        request.body().writeTo(buffer);
        var body = JSONUtil.parseObj(buffer.readUtf8());
        assertThat(request.url()).isEqualTo("https://example.com/v1/images/generations");
        assertThat(body.getStr("model")).isEqualTo(code);
        assertThat(body.getStr("size")).isEqualTo(expectedSize);
        assertThat(body.getInt("n")).isEqualTo(1);
    }

    @ParameterizedTest
    @CsvSource({
            "qwen-image-3.0-pro,2K,16:9,2688x1536",
            "qwen-image-2.0,2K,9:16,1536x2688",
            "jimeng-image-4.6,1K,4:3,1216x912"
    })
    void imageOptionsProduceDimensionsWithinThePublishedPixelBounds(String code, String tier,
                                                                    String ratio, String expectedSize) throws Exception {
        var config = JSONUtil.parseObj(PRESETS.getPresetConfig("infistar:" + code));
        var support = new OpenAiCompatibleImageProtocolSupport(PRESETS, mock(StorageConfigService.class),
                new PresetArtStyleResourceResolver());
        var task = ImageTask.builder().prompt("海景").resolution(tier).ratio(ratio).build();
        int[] size = support.resolveConfiguredSize(task, config);
        var context = new OpenAiCompatibleImageProtocolContext(AiModel.builder().code(code).build(), api,
                code, task.getPrompt(), size[0], size[1], 1, List.of(), config, false);
        var buffer = new Buffer();
        new OpenAiImageProtocolAdapter(support).buildRequest(context).body().writeTo(buffer);
        assertThat(JSONUtil.parseObj(buffer.readUtf8()).getStr("size")).isEqualTo(expectedSize);
    }

    @ParameterizedTest
    @CsvSource({
            "veo-3.1-fast-generate-preview,8,720p,16:9,1280x720",
            "veo-3.1-fast-generate-preview,8,1080p,9:16,1080x1920",
            "doubao-seedance-2.0-mini,5,720p,1:1,720x720",
            "viduq3-pro,5,540p,16:9,960x540",
            "viduq3-turbo,5,1080p,4:3,1440x1080",
            "wan3.0-video,5,1080p,9:16,1080x1920"
    })
    void videoResolutionAndRatioReachTheProvider(String code, int seconds, String resolution,
                                                 String ratio, String expectedSize) throws Exception {
        var config = JSONUtil.parseObj(PRESETS.getPresetConfig("infistar:" + code));
        var task = VideoTask.builder().prompt("海边的灯塔").resolution(resolution).ratio(ratio).build();
        VideoModelRules.normalizeTask(config, task);
        var context = new OpenAiCompatibleVideoProtocolContext(AiModel.builder().code(code).build(), api, task, config, null);
        var support = new OpenAiCompatibleVideoProtocolSupport(mock(StorageConfigService.class),
                mock(PresetArtStyleResourceResolver.class));
        var buffer = new Buffer();
        new InfistarVideoProtocolAdapter(support).buildSubmitBody(context).writeTo(buffer);
        assertThat(task.getDuration()).isEqualTo(seconds);
        assertThat(buffer.readUtf8()).containsPattern("name=\"size\"\r\n(?:[^\r\n]+\r\n)*\r\n" + expectedSize + "\r\n")
                .containsPattern("name=\"seconds\"\r\n(?:[^\r\n]+\r\n)*\r\n" + seconds + "\r\n");
    }

    @Test
    void veoRejectsFiveSecondsAndHighResolutionShortClips() {
        var config = JSONUtil.parseObj(PRESETS.getPresetConfig("infistar:veo-3.1-fast-generate-preview"));
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config,
                VideoTask.builder().prompt("海浪").duration(5).build())).hasMessageContaining("时长");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config,
                VideoTask.builder().prompt("海浪").duration(6).resolution("1080p").build())).hasMessageContaining("1080p");
        var supported = VideoTask.builder().prompt("海浪").duration(6).resolution("720p").build();
        VideoModelRules.normalizeTask(config, supported);
        assertThat(supported.getDuration()).isEqualTo(6);
    }

    @Test
    void genericDefaultsStillKeepTheExactFutureModelIdAndARealSize() throws Exception {
        var config = JSONUtil.parseObj(PRESETS.getPresetConfig("infistar:common-video"));
        var task = VideoTask.builder().prompt("海浪").build();
        VideoModelRules.normalizeTask(config, task);
        var context = new OpenAiCompatibleVideoProtocolContext(AiModel.builder().code("future-video").build(), api, task, config, null);
        var buffer = new Buffer();
        new InfistarVideoProtocolAdapter(new OpenAiCompatibleVideoProtocolSupport(mock(StorageConfigService.class),
                mock(PresetArtStyleResourceResolver.class))).buildSubmitBody(context).writeTo(buffer);
        assertThat(buffer.readUtf8()).contains("future-video", "1280x720");
        assertThat(task.getDuration()).isEqualTo(5);
    }
}
