package com.stonewu.fusion.controller.ai.vo;

import com.stonewu.fusion.common.PageParam;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Schema(description = "API配置分页查询请求")
@Data
@EqualsAndHashCode(callSuper = true)
public class ApiConfigPageReqVO extends PageParam {
    private String name;
    private String platform;
    private Integer status;
}
