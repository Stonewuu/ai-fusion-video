package com.stonewu.fusion.controller.storyboard.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

@Schema(description = "创建分镜场次请求")
@Data
public class StoryboardSceneCreateReqVO {

    private Long episodeId;

    private Long storyboardId;

    private String sceneNumber;

    private String sceneHeading;

    private String location;

    private String timeOfDay;

    private String intExt;

    private Integer sortOrder;
}
