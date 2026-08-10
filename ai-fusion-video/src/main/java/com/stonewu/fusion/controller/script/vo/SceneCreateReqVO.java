package com.stonewu.fusion.controller.script.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 创建场次请求 VO
 */
@Schema(description = "创建场次请求")
@Data
public class SceneCreateReqVO {

    @NotNull(message = "分集ID不能为空")
    private Long episodeId;

    private Long scriptId;

    private String sceneNumber;

    private String sceneHeading;

    private String location;

    private String timeOfDay;

    private String intExt;

    private String characters;

    private String characterAssetIds;

    private Long sceneAssetId;

    private String propAssetIds;

    private String sceneDescription;

    private String dialogues;

    private Integer sortOrder;
}
