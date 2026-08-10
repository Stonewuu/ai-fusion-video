package com.stonewu.fusion.controller.team.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "更新团队请求")
@Data
public class TeamUpdateReqVO {
    @NotNull(message = "团队ID不能为空")
    private Long id;
    private String name;
    private String description;
    private String logo;
    private Integer status;
}
