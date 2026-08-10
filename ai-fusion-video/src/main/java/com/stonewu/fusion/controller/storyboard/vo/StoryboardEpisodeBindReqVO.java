package com.stonewu.fusion.controller.storyboard.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

/**
 * 绑定分镜集和剧本分集请求。
 */
@Schema(description = "绑定分镜集和剧本分集请求")
@Data
public class StoryboardEpisodeBindReqVO {

    /** 剧本分集ID */
    @NotNull(message = "剧本分集ID不能为空")
    private Long scriptEpisodeId;
}
