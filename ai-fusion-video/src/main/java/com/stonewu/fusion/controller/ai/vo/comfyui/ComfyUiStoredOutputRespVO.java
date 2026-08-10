package com.stonewu.fusion.controller.ai.vo.comfyui;

import lombok.Builder;
import lombok.Value;

@Value
@Builder
public class ComfyUiStoredOutputRespVO {

    String mediaType;
    String role;
    String url;
    long size;
}
