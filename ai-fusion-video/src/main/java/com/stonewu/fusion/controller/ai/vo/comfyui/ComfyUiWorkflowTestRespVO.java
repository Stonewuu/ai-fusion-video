package com.stonewu.fusion.controller.ai.vo.comfyui;

import lombok.Builder;
import lombok.Value;

import java.util.List;

@Value
@Builder
public class ComfyUiWorkflowTestRespVO {

    boolean passed;
    String promptId;
    long durationMillis;
    List<ComfyUiStoredOutputRespVO> outputs;
    String message;
}
