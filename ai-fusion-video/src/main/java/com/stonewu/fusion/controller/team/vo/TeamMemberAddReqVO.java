package com.stonewu.fusion.controller.team.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Schema(description = "添加团队成员请求")
@Data
public class TeamMemberAddReqVO {
    @NotNull(message = "团队ID不能为空")
    private Long teamId;
    @NotNull(message = "用户ID不能为空")
    private Long userId;
    private Integer role;
}
