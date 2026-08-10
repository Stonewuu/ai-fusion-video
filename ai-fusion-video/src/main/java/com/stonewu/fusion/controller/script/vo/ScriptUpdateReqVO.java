package com.stonewu.fusion.controller.script.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 更新剧本请求 VO
 */
@Schema(description = "更新剧本请求")
@Data
public class ScriptUpdateReqVO {

    @NotNull(message = "剧本ID不能为空")
    private Long id;

    private String content;

    private String rawContent;

    private String storySynopsis;

    private String genre;

    private String targetAudience;

    private Integer durationEstimate;

    /** 乐观锁版本号 */
    private Integer version;
}
