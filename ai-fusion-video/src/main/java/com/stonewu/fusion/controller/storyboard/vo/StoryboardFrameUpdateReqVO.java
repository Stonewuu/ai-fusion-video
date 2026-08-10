package com.stonewu.fusion.controller.storyboard.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * 分镜首尾帧手动更新请求 VO。
 */
@Schema(description = "分镜首尾帧手动更新请求")
@Data
public class StoryboardFrameUpdateReqVO {

    @Schema(description = "帧类型：first-首帧，last-尾帧", requiredMode = Schema.RequiredMode.REQUIRED)
    @NotBlank(message = "帧类型不能为空")
    private String frameType;

    @Schema(description = "图片URL，传空表示清空")
    private String imageUrl;

    @Schema(description = "帧生成提示词，手动上传时可为空")
    private String prompt;
}
