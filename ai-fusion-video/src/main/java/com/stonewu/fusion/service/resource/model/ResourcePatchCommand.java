package com.stonewu.fusion.service.resource.model;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

public record ResourcePatchCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @NotNull CanvasValueType resourceType,
        @Positive long resourceId,
        @NotNull @PositiveOrZero Integer expectedVersion,
        @NotNull ObjectNode patch,
        @PositiveOrZero Integer expectedParentVersion) implements ResourceCommand {
    public ResourcePatchCommand(String requestId, CanvasValueType resourceType, long resourceId,
                                Integer expectedVersion, ObjectNode patch) {
        this(requestId, resourceType, resourceId, expectedVersion, patch, null);
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.PATCH;
    }
}
