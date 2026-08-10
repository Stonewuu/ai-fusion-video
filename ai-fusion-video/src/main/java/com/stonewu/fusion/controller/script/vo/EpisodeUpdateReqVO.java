package com.stonewu.fusion.controller.script.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 更新分集请求 VO
 */
@Schema(description = "更新分集请求")
@Data
public class EpisodeUpdateReqVO {

    @NotNull(message = "分集ID不能为空")
    private Long id;

    private String title;

    private String synopsis;

    private String rawContent;

    private Integer durationEstimate;

    private Integer sortOrder;

    /** 乐观锁版本号 */
    private Integer version;
}
