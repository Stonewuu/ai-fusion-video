package com.stonewu.fusion.controller.system.vo;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * 发起重置密码请求 DTO
 */
@Data
public class PasswordResetRequestVO {

    @NotBlank(message = "用户名或邮箱不能为空")
    private String usernameOrEmail;

    @NotBlank(message = "找回方式不能为空")
    private String type; // email 或是 log
}
