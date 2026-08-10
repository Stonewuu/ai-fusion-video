package com.stonewu.fusion.controller.team.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import java.time.LocalDateTime;

@Schema(description = "团队成员响应")
@Data
public class TeamMemberRespVO {
    private Long id;
    private Long teamId;
    private Long userId;
    private Integer role;
    private Integer status;
    private LocalDateTime joinTime;
}
