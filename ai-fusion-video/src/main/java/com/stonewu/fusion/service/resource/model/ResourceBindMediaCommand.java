package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

import java.util.Map;

public record ResourceBindMediaCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @NotNull CanvasValueType resourceType,
        @Positive long resourceId,
        @NotNull @PositiveOrZero Integer expectedVersion,
        @PositiveOrZero Integer expectedParentVersion,
        @NotNull @Size(min = 1, max = 12) Map<@NotBlank String, @NotBlank String> bindings) implements ResourceCommand {
    @Override
    public ResourceOperation operation() {
        return ResourceOperation.BIND_MEDIA;
    }
}
