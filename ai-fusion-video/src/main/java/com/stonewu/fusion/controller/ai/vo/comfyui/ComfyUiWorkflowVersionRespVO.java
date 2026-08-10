package com.stonewu.fusion.controller.ai.vo.comfyui;

import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ComfyUiWorkflowVersionRespVO {

    private Long id;
    private Long workflowId;
    private Integer versionNo;
    private String uiWorkflowJson;
    private String apiWorkflowJson;
    private String inputBindingsJson;
    private String outputBindingsJson;
    private String requiredNodesJson;
    private String workflowHash;
    private Integer validationStatus;
    private String validationMessage;
    private Integer testStatus;
    private String testMessage;
    private LocalDateTime lastTestTime;
    private Boolean published;
    private LocalDateTime createTime;
    private LocalDateTime updateTime;
}
