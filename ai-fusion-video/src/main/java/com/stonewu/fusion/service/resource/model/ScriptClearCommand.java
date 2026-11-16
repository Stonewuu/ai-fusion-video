package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.PositiveOrZero;
import jakarta.validation.constraints.Size;

public record ScriptClearCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @Positive long resourceId,
        @NotNull @PositiveOrZero Integer expectedVersion) implements ResourceCommand {
    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.SCRIPT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.CLEAR_SCRIPT_CONTENT;
    }
}
