package com.stonewu.fusion.service.ai.comfyui;

/** One output file descriptor resolved from a bound ComfyUI node. */
public record ComfyUiRemoteOutput(
        String nodeId,
        String mediaType,
        String role,
        String filename,
        String subfolder,
        String type) {
}
