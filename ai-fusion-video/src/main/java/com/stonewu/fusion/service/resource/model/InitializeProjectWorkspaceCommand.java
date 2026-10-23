package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

public record InitializeProjectWorkspaceCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @Positive long resourceId,
        @NotNull @PositiveOrZero Integer expectedVersion) implements ResourceCommand {
    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.PROJECT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.INITIALIZE_WORKSPACE;
    }
}
