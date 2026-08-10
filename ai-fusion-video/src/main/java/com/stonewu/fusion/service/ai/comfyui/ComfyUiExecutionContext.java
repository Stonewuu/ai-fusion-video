package com.stonewu.fusion.service.ai.comfyui;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;

/** Fully resolved immutable context for one ComfyUI execution. */
public record ComfyUiExecutionContext(
        AiModel model,
        ApiConfig apiConfig,
        ComfyUiWorkflow workflow,
        ComfyUiWorkflowVersion version) {
}
