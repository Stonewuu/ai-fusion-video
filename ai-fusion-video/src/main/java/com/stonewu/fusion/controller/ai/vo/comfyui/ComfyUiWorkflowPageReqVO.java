package com.stonewu.fusion.controller.ai.vo.comfyui;

import com.stonewu.fusion.common.PageParam;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = true)
public class ComfyUiWorkflowPageReqVO extends PageParam {

    private String name;

    private String code;

    private Long apiConfigId;

    private Integer modelType;

    private Integer status;
}
