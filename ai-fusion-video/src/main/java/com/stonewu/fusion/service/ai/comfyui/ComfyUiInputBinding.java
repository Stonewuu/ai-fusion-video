package com.stonewu.fusion.service.ai.comfyui;

/**
 * One platform input mapped to a concrete ComfyUI node input.
 */
public record ComfyUiInputBinding(
        String businessField,
        String nodeId,
        String inputName,
        String valueType,
        Integer index,
        boolean optional) {

    public ComfyUiInputBinding(String businessField, String nodeId, String inputName, String valueType, Integer index) {
        this(businessField, nodeId, inputName, valueType, index, false);
    }
}
