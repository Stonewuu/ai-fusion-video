package com.stonewu.fusion.controller.generation.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * 提交生视频任务请求 VO
 */
@Schema(description = "提交生视频任务请求")
@Data
public class VideoTaskSubmitReqVO {

    private Long projectId;

    @NotBlank(message = "提示词不能为空")
    private String prompt;

    private Long promptTemplateId;

    private String generateMode;

    private String firstFrameImageUrl;

    private String lastFrameImageUrl;

    private String referenceImageUrls;

    private String referenceVideoUrls;

    private String referenceAudioUrls;

    private String ratio;

    private String resolution;

    private Integer duration;

    private Boolean watermark;

    private Boolean generateAudio;

    private Long seed;

    private Boolean cameraFixed;

    private Integer count;

    private Long modelId;

    private String category;
}
