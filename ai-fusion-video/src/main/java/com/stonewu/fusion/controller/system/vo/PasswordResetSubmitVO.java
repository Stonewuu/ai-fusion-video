package com.stonewu.fusion.controller.system.vo;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * 提交重置密码 DTO
 */
@Data
public class PasswordResetSubmitVO {

    @NotBlank(message = "重置凭证不能为空")
    private String token; // 邮箱 token 或是日志验证码

    @NotBlank(message = "找回方式不能为空")
    private String type; // email 或是 log

    @NotBlank(message = "新密码不能为空")
    private String newPassword;

    @NotBlank(message = "确认新密码不能为空")
    private String confirmPassword;
}
