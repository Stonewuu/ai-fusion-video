package com.stonewu.fusion.controller.system.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import java.time.LocalDateTime;

@Schema(description = "角色响应")
@Data
public class RoleRespVO {
    private Long id;
    private String name;
    private String code;
    private Integer sort;
    private Integer status;
    private String remark;
    private LocalDateTime createTime;
}
