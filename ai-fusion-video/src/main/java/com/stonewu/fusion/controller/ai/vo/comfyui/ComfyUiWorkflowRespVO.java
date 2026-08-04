package com.stonewu.fusion.controller.ai.vo.comfyui;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ComfyUiWorkflowRespVO {

    private Long id;
    private Long apiConfigId;
    private String name;
    private String code;
    private Integer modelType;
    private String description;
    private Long activeVersionId;
    private Integer status;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
