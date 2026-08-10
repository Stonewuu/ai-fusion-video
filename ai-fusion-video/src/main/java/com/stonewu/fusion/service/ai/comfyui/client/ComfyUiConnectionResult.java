package com.stonewu.fusion.service.ai.comfyui.client;

import com.fasterxml.jackson.databind.JsonNode;

/** Result of validating a configured ComfyUI Native API endpoint. */
public record ComfyUiConnectionResult(
        boolean connected,
        boolean jobsApiSupported,
        String version,
        JsonNode systemStats,
        JsonNode features) {
}
