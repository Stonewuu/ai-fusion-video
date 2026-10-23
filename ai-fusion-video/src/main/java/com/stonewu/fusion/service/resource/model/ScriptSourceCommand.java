package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

public record ScriptSourceCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @Positive long resourceId,
        @NotNull @PositiveOrZero Integer expectedVersion,
        @NotBlank @Size(max = 2_000_000) String rawContent) implements ResourceCommand {
    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.SCRIPT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.REPLACE_SCRIPT_SOURCE;
    }
}
