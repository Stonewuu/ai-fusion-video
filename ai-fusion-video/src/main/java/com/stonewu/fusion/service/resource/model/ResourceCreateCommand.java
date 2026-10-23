package com.stonewu.fusion.service.resource.model;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

public record ResourceCreateCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @NotNull CanvasValueType resourceType,
        @Positive Long projectId,
        @NotNull ObjectNode fields) implements ResourceCommand {
    @Override
    public ResourceOperation operation() {
        return ResourceOperation.CREATE;
    }
}
