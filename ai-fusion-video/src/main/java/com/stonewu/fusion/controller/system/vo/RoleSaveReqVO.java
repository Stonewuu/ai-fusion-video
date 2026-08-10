package com.stonewu.fusion.controller.system.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Schema(description = "角色保存请求")
@Data
public class RoleSaveReqVO {
    private Long id;
    @NotBlank(message = "角色名称不能为空")
    private String name;
    @NotBlank(message = "角色编码不能为空")
    private String code;
    private Integer sort;
    private String remark;
}
