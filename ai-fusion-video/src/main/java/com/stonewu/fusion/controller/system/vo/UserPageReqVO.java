package com.stonewu.fusion.controller.system.vo;

import com.stonewu.fusion.common.PageParam;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "用户分页查询请求")
@Data
@EqualsAndHashCode(callSuper = true)
public class UserPageReqVO extends PageParam {
    private String username;
    private String nickname;
    private Integer status;
}
