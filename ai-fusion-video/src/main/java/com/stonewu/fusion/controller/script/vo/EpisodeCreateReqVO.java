package com.stonewu.fusion.controller.script.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 创建分集请求 VO
 */
@Schema(description = "创建分集请求")
@Data
public class EpisodeCreateReqVO {

    @NotNull(message = "剧本ID不能为空")
    private Long scriptId;

    private Integer episodeNumber;

    private String title;

    private String synopsis;

    private String rawContent;

    private Integer durationEstimate;

    private Integer sortOrder;
}
