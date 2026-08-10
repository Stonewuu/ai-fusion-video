package com.stonewu.fusion.controller.storyboard.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.math.BigDecimal;

/**
 * 创建分镜条目请求 VO
 */
@Schema(description = "创建分镜条目请求")
@Data
public class StoryboardItemCreateReqVO {

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

    private String characterIds;

    private Long sceneId;

    private Long sceneAssetItemId;

    private String propIds;

    private String remark;

    private String customData;
}
