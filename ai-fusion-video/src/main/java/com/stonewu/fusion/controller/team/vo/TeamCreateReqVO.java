package com.stonewu.fusion.controller.team.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Schema(description = "创建团队请求")
@Data
public class TeamCreateReqVO {
    @NotBlank(message = "团队名称不能为空")
    private String name;
    private String description;
}
