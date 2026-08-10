package com.stonewu.fusion.controller.ai.vo.comfyui;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.Builder;
import lombok.Value;

@Value
@Builder
public class ComfyUiConnectionRespVO {

    boolean connected;
    boolean jobsApiSupported;
    String version;
    JsonNode systemStats;
    JsonNode features;
}
