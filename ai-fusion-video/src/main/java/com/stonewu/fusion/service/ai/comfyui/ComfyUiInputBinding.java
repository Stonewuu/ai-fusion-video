package com.stonewu.fusion.service.ai.comfyui;

/** One platform input mapped to a concrete ComfyUI node input. */
public record ComfyUiInputBinding(
        String businessField,
        String nodeId,
        String inputName,
        String valueType,
        Integer index) {
}
