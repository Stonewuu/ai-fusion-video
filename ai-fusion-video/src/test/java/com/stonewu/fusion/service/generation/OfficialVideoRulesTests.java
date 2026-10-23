package com.stonewu.fusion.service.generation;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.media.model.MediaFacts;
import com.stonewu.fusion.service.media.model.MediaKind;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.mock;

/**
 * 使用产品预设，而非测试专用规则；完全不启动 Spring 或数据库。
 */
class OfficialVideoRulesTests {
    private static final ModelPresetService PRESETS = new ModelPresetService();
    private final GenerationModelCapabilityService capabilities = new GenerationModelCapabilityService(
            mock(AiModelMetadataResolver.class), PRESETS);

    @BeforeAll
    static void load() {
        PRESETS.init();
    }

    private JSONObject config(String code) {
        return JSONUtil.parseObj(PRESETS.getPresetConfig(code));
    }

    private AiModel model(String code) {
        return AiModel.builder().code(code).config(config(code).toString()).build();
    }

    private VideoTask task() {
        return VideoTask.builder().prompt("海边灯塔，镜头缓慢向前移动").build();
    }

    @Test
    void automaticDurationIsNotNegativeBudgetOrFixedDuration() {
        for (var code : List.of("doubao-seedance-2-5-260628", "wan3.0-video")) {
            var task = task();
            task.setDuration(-1);
            VideoModelRules.normalizeTask(config(code), task);
            assertThat(task.getDuration()).isEqualTo(-1);
            assertThat(VideoModelRules.budgetSeconds(config(code), task.getDuration())).isEqualTo(30);
        }
        var defaultTask = task();
        VideoModelRules.normalizeTask(config("doubao-seedance-2-5-260628"), defaultTask);
        assertThat(defaultTask.getDuration()).isEqualTo(5);
        assertThat(VideoModelRules.budgetSeconds(config("doubao-seedance-2-5-260628"), null)).isEqualTo(5);
        var task = task();
        task.setDuration(-1);
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config("MiniMax-H3"), task)).hasMessageContaining(
                "不支持自动时长");
    }

    /** 快速配置的星河万相从已验证的五秒开始，短时长须在调用付费接口前被拒绝。 */
    @Test
    void infistarWanRejectsUnsupportedShortDurationsWithoutChangingOfficialWan() {
        for (var code : List.of("infistar:wan3.0-video", "infistar:wan3.0-video-prime")) {
            var model = AiModel.builder().code(code.substring("infistar:".length()))
                    .capabilityPresetCode(code).modelType(3).build();
            var task = task();
            capabilities.validateVideoTask(model, task, "openai_compatible");
            assertThat(task.getDuration()).isEqualTo(5);
            for (int duration = 2; duration < 5; duration++) {
                task.setDuration(duration);
                assertThatThrownBy(() -> capabilities.validateVideoTask(model, task, "openai_compatible"))
                        .hasMessageContaining("5 至 30 秒");
            }
        }
        var officialTask = task();
        officialTask.setDuration(2);
        VideoModelRules.normalizeTask(config("wan3.0-video"), officialTask);
        assertThat(officialTask.getDuration()).isEqualTo(2);
    }

    @Test
    void h3VariantsHaveDifferentResolutionsAndDurationsAndNativeAudio() {
        var high = task();
        high.setResolution("2k");
        high.setDuration(4);
        VideoModelRules.normalizeTask(config("MiniMax-H3"), high);
        assertThat(high.getResolution()).isEqualTo("2K");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config("MiniMax-H3-Max"), high)).hasMessageContaining(
                "分辨率");
        high.setResolution("768P");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config("MiniMax-H3-Max"), high)).hasMessageContaining(
                "时长");
        high.setDuration(5);
        high.setGenerateAudio(false);
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config("MiniMax-H3"), high)).hasMessageContaining(
                "不能关闭声音");
    }

    @Test
    void promptAndRatioRulesDependOnInputRatherThanModelNameGuesses() {
        var h3 = task();
        h3.setRatio("adaptive");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config("MiniMax-H3"), h3)).hasMessageContaining(
                "纯文本");
        h3.setFirstFrameImageUrl("https://example.com/frame.png");
        h3.setRatio("16:9");
        VideoModelRules.normalizeTask(config("MiniMax-H3"), h3);
        assertThat(h3.getRatio()).isEqualTo("adaptive");
        h3.setPrompt("");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config("MiniMax-H3"), h3)).hasMessageContaining(
                "创作要求");
        var seed = task();
        seed.setPrompt("");
        seed.setReferenceAudioUrls("[\"https://example.com/sound.mp3\"]");
        VideoModelRules.normalizeTask(config("doubao-seedance-2-5-260628"), seed);
        assertThat(seed.getPrompt()).isEmpty();
        seed.setReferenceAudioUrls(null);
        assertThatThrownBy(
                () -> VideoModelRules.normalizeTask(config("doubao-seedance-2-5-260628"), seed)).hasMessageContaining(
                "参考素材");
    }

    @Test
    void inputCountsAndMutuallyExclusiveModesAreCheckedBeforeSubmission() {
        var task = task();
        task.setFirstFrameImageUrl("https://example.com/first.png");
        task.setReferenceVideoUrls("[\"https://example.com/ref.mp4\"]");
        assertThatThrownBy(
                () -> capabilities.validateVideoTask(model("wan3.0-video"), task, "dashscope")).hasMessageContaining(
                "不能混用");
        task.setFirstFrameImageUrl(null);
        task.setReferenceVideoUrls(JSONUtil.toJsonStr(List.of("a", "b", "c", "d")));
        assertThatThrownBy(() -> capabilities.validateVideoTask(model("MiniMax-H3"), task,
                "openai_compatible")).hasMessageContaining("3");
        task.setReferenceVideoUrls(null);
        task.setLastFrameImageUrl("https://example.com/last.png");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config("wan3.0-video"), task)).hasMessageContaining(
                "首帧");
    }

    @Test
    void promptCountsUnicodeCharactersAndInvalidParametersStayVisible() {
        var task = task();
        task.setPrompt("🌅".repeat(7000));
        VideoModelRules.normalizeTask(config("MiniMax-H3"), task);
        task.setPrompt(task.getPrompt() + "。");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config("MiniMax-H3"), task)).hasMessageContaining(
                "7000");
        task.setPrompt("海浪");
        task.setSeed(4L);
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config("MiniMax-H3"), task)).hasMessageContaining(
                "随机种子");
        task.setSeed(2147483648L);
        task.setResolution(null);
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(config("wan3.0-video"), task)).hasMessageContaining(
                "2147483647");
    }

    @Test
    void advancedOptionsAreExplicitAndTyped() {
        var cfg = config("MiniMax-H3");
        cfg.set("promptExpansionMode", "balanced");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(cfg, task())).hasMessageContaining("提示词优化");
        var max = config("MiniMax-H3-Max");
        max.set("promptExpansionMode", "balance");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(max, task())).hasMessageContaining("提示词优化");
        var seed = config("doubao-seedance-2-5-260628");
        seed.set("referenceTaskType", "edit");
        var task = task();
        task.setReferenceVideoUrls("[\"https://example.com/input.mp4\"]");
        task.setDuration(5);
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(seed, task)).hasMessageContaining("自动时长");
        task.setDuration(-1);
        VideoModelRules.normalizeTask(seed, task);
        var wan = config("wan3.0-video");
        wan.set("promptExtend", "true");
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(wan, task())).hasMessageContaining("扩写");
    }

    @Test
    void probedMediaLimitsIncludeAggregateAndWanInputPlusOutput() {
        var clip = video(8000, 1280, 720);
        assertThatThrownBy(
                () -> VideoMediaRules.validate(config("MiniMax-H3"), 5, List.of(), List.of(clip, clip), List.of()))
                .hasMessageContaining("总时长");
        assertThatThrownBy(
                () -> VideoMediaRules.validate(config("wan3.0-video"), 25, List.of(), List.of(clip), List.of()))
                .hasMessageContaining("合计");
        VideoMediaRules.validate(config("wan3.0-video"), -1, List.of(), List.of(clip), List.of());
        assertThatThrownBy(
                () -> VideoMediaRules.validate(config("MiniMax-H3"), 5, List.of(), List.of(video(1000, 1280, 720)),
                        List.of()))
                .hasMessageContaining("每段时长");
        assertThatThrownBy(() -> VideoMediaRules.validate(config("doubao-seedance-2-5-260628"), 5, List.of(),
                List.of(video(5000, 300, 300)), List.of()))
                .hasMessageContaining("总像素");
    }

    @Test
    void oldProfilesDoNotAcquireNewConstraintsAndMalformedContractsFail() {
        var task = task();
        task.setDuration(5);
        task.setRatio("16:9");
        var old = config("doubao-seedance-2-0-260128");
        assertThat(VideoModelRules.from(old)).isNull();
        VideoModelRules.normalizeTask(old, task);
        assertThat(task.getResolution()).isNull();
        var invalid = config("MiniMax-H3");
        invalid.getJSONObject("videoRules").set("version", 9);
        assertThatThrownBy(() -> VideoModelRules.normalizeTask(invalid, task)).isInstanceOf(BusinessException.class)
                .hasMessageContaining("版本");
    }

    private MediaFacts video(long millis, int width, int height) {
        return new MediaFacts(MediaKind.VIDEO, "video/mp4", 1024, "hash", width, height, millis, true);
    }
}
