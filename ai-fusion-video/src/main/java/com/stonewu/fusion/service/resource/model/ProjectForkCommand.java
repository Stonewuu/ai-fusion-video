package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record ProjectForkCommand(@NotBlank @Pattern(regexp = "[A-Za-z0-9:._-]{1,96}") String requestId,
                                 @NotBlank @Pattern(regexp = "[0-9a-f-]{36}") String sourceCanvasId,
                                 @Min(1) long expectedRevision,
                                 @NotBlank @Pattern(regexp = "[0-9a-f]{64}") String digest,
                                 @NotBlank @Pattern(regexp = "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}") String canvasId,
                                 @NotBlank @Size(max = 200) String title) implements ResourceCommand {
    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.PROJECT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.FORK_PROJECT;
    }
}
