package com.stonewu.fusion.controller.storyboard.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotEmpty;
import lombok.Data;

import java.util.List;

@Schema(description = "批量更新分镜条目排序请求")
@Data
public class StoryboardItemSortReqVO {

    @NotEmpty(message = "分镜条目ID列表不能为空")
    @Schema(description = "分镜条目ID列表(按顺序)", requiredMode = Schema.RequiredMode.REQUIRED)
    private List<Long> ids;
}
