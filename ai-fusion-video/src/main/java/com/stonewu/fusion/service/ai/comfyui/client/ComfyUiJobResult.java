package com.stonewu.fusion.service.ai.comfyui.client;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

/** Normalized response from ComfyUI /api/jobs/{id}. */
public record ComfyUiJobResult(
        String id,
        String status,
        ObjectNode outputs,
        JsonNode executionError,
        JsonNode raw) {

    public boolean completed() {
        return "completed".equals(status);
    }

    public boolean failed() {
        return "failed".equals(status) || "cancelled".equals(status);
    }
}
