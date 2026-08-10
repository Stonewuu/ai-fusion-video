package com.stonewu.fusion.service.ai.comfyui.client;

import com.fasterxml.jackson.databind.JsonNode;

/** Accepted ComfyUI prompt queue response. */
public record ComfyUiPromptResponse(
        String promptId,
        Double number,
        JsonNode nodeErrors) {
}
