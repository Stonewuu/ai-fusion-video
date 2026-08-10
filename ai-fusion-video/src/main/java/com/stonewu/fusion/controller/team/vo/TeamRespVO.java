package com.stonewu.fusion.controller.team.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import java.time.LocalDateTime;

@Schema(description = "团队响应")
@Data
public class TeamRespVO {
    private Long id;
    private String name;
    private String logo;
    private String description;
    private Long ownerUserId;
    private Integer status;
    private Long memberCount;
    private LocalDateTime createTime;
}
