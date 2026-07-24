package com.stonewu.fusion.controller.storyboard.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

import java.math.BigDecimal;

/**
 * 更新分镜条目请求 VO
 */
@Schema(description = "更新分镜条目请求")
@Data
public class StoryboardItemUpdateReqVO {

    @NotNull(message = "分镜条目ID不能为空")
    private Long id;

    private Long storyboardId;

    private Long storyboardEpisodeId;

    private Long storyboardSceneId;

    private Integer sortOrder;

    private String shotNumber;

    private String autoShotNumber;

    private String imageUrl;

    private String referenceImageUrl;

    private String videoUrl;

    private String generatedImageUrl;

    private String firstFrameImageUrl;

    private String lastFrameImageUrl;

    private String firstFramePrompt;

    private String lastFramePrompt;

    private String generatedVideoUrl;

    private String videoPrompt;

    private String shotType;


    private BigDecimal duration;

    private String content;

    private String sceneExpectation;

    private String sound;

    private String dialogue;

    private String soundEffect;

    private String music;

    private String cameraMovement;

    private String cameraAngle;

    private String cameraEquipment;

    private String focalLength;

    private String transition;

    private Long sceneId;

    private String remark;

    private String customData;
}
