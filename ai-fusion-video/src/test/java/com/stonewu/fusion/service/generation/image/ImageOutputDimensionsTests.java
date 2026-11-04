package com.stonewu.fusion.service.generation.image;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.ModelPresetService;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ImageOutputDimensionsTests {
    private static final ModelPresetService presets = new ModelPresetService();

    @BeforeAll
    static void loadPresets() {
        presets.init();
    }

    @ParameterizedTest
    @CsvSource({
            "pro/gpt-image-2.5-sunburst,1K,9:16,864,1536",
            "gpt-image-2.5-flare-2026-09-08,4K,1:1,2880,2880",
            "pro/gpt-image-2-2026-04-21,2K,9:16,1152,2048"
    })
    void persistsTheSelectedTierAndPixelSizeForRoutedModels(String model, String tier, String ratio, int width, int height) {
        var config = JSONUtil.parseObj(presets.getPresetConfig(presets.findPresetCode(model, 2)));
        var task = ImageTask.builder().resolution(tier).ratio(ratio).build();
        ImageOutputDimensions.normalize(task, config, true);
        assertThat(task.getWidth()).isEqualTo(width);
        assertThat(task.getHeight()).isEqualTo(height);
        assertThat(task.getResolution()).isEqualTo(tier);
        assertThat(task.getRatio()).isEqualTo(ratio);
    }

    @Test
    void rejectsLandscapePixelsForAPortraitRequestButAcceptsRoundedLegacyPixels() {
        var config = JSONUtil.parseObj(presets.getPresetConfig("gpt-image-2.5-sunburst"));
        var conflict = ImageTask.builder().resolution("2048x1024").ratio("9:16").build();
        assertThatThrownBy(() -> ImageOutputDimensions.normalize(conflict, config, true))
                .isInstanceOf(BusinessException.class).hasMessageContaining("宽高与画面比例不一致");
        var legacy = ImageTask.builder().resolution("2048x1360").ratio("3:2").build();
        ImageOutputDimensions.normalize(legacy, config, true);
        assertThat(legacy.getWidth()).isEqualTo(2048);
        assertThat(legacy.getHeight()).isEqualTo(1360);
        var blank = new ImageTask();
        ImageOutputDimensions.normalize(blank, new JSONObject(), true);
        assertThat(blank.getResolution()).isNull();
    }

    @ParameterizedTest
    @ValueSource(strings = {"1025x1024", "3840x960", "3840x3840", "512x512"})
    void rejectsInvalidOpenAiDimensionsBeforeSubmission(String size) {
        var task = ImageTask.builder().resolution(size).build();
        assertThatThrownBy(() -> ImageOutputDimensions.normalize(task, new JSONObject(), true))
                .isInstanceOf(BusinessException.class).hasMessageContaining("总像素");
    }
}
