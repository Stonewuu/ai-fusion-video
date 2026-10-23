package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

public record ResourceDeleteCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @NotNull CanvasValueType resourceType,
        @Positive long resourceId,
        @NotNull @PositiveOrZero Integer expectedVersion,
        @PositiveOrZero Integer expectedParentVersion) implements ResourceCommand {
    public ResourceDeleteCommand(String requestId, CanvasValueType resourceType, long resourceId,
                                 Integer expectedVersion) {
        this(requestId, resourceType, resourceId, expectedVersion, null);
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.DELETE;
    }
}
