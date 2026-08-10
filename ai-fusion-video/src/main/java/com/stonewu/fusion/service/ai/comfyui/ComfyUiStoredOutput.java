package com.stonewu.fusion.service.ai.comfyui;

/** Platform-persisted output created from a ComfyUI result. */
public record ComfyUiStoredOutput(
        String mediaType,
        String role,
        String url,
        long size) {
}
