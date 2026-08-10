package com.stonewu.fusion.controller.system.vo;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import java.time.LocalDateTime;
import java.util.List;

@Schema(description = "用户信息响应")
@Data
public class UserRespVO {
    private Long id;
    private String username;
    private String nickname;
    private String avatar;
    private String email;
    private String phone;
    private Integer status;
    private LocalDateTime createTime;
    private List<String> roles;
}
