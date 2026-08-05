package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.node.ObjectNode;

/** Rendered prompt with a platform-generated recoverable UUID. */
public record ComfyUiPreparedSubmission(
        ComfyUiExecutionContext context,
        String promptId,
        ObjectNode prompt) {
}
