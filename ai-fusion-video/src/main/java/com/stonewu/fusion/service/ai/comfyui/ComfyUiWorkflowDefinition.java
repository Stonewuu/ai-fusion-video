package com.stonewu.fusion.service.ai.comfyui;

import java.util.List;

/** Canonicalized and locally validated workflow definition. */
public record ComfyUiWorkflowDefinition(
        String uiWorkflowJson,
        String apiWorkflowJson,
        String inputBindingsJson,
        String outputBindingsJson,
        String requiredNodesJson,
        String workflowHash,
        List<ComfyUiInputBinding> inputBindings,
        List<ComfyUiOutputBinding> outputBindings) {
}
