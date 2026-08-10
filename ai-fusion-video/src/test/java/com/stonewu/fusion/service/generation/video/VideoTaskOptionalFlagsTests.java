package com.stonewu.fusion.service.generation.video;

import com.stonewu.fusion.controller.generation.vo.VideoTaskSubmitReqVO;
import com.stonewu.fusion.convert.generation.GenerationConvert;
import com.stonewu.fusion.entity.generation.VideoTask;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class VideoTaskOptionalFlagsTests {

    @Test
    void builderLeavesOptionalFlagsUnsetWhenCallerOmitsThem() {
        VideoTask task = VideoTask.builder().build();

        assertThat(task.getWatermark()).isNull();
        assertThat(task.getGenerateAudio()).isNull();
        assertThat(task.getCameraFixed()).isNull();
    }

    @Test
    void convertKeepsUnsetOptionalFlagsNull() {
        VideoTaskSubmitReqVO reqVO = new VideoTaskSubmitReqVO();
        reqVO.setPrompt("test prompt");

        VideoTask task = GenerationConvert.INSTANCE.convert(reqVO);

        assertThat(task.getWatermark()).isNull();
        assertThat(task.getGenerateAudio()).isNull();
        assertThat(task.getCameraFixed()).isNull();
    }

    @Test
    void convertPreservesExplicitFalseForOptionalFlags() {
        VideoTaskSubmitReqVO reqVO = new VideoTaskSubmitReqVO();
        reqVO.setPrompt("test prompt");
        reqVO.setWatermark(false);
        reqVO.setGenerateAudio(false);
        reqVO.setCameraFixed(false);

        VideoTask task = GenerationConvert.INSTANCE.convert(reqVO);

        assertThat(task.getWatermark()).isFalse();
        assertThat(task.getGenerateAudio()).isFalse();
        assertThat(task.getCameraFixed()).isFalse();
    }
}