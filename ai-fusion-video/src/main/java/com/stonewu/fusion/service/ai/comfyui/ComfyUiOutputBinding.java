package com.stonewu.fusion.service.ai.comfyui;

/** Explicit selection of one ComfyUI output node. */
public record ComfyUiOutputBinding(
        String nodeId,
        String mediaType,
        String role) {
}
