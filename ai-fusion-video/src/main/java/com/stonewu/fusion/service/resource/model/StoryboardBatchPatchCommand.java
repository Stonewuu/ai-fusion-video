package com.stonewu.fusion.service.resource.model;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;

import java.util.List;

public record StoryboardBatchPatchCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @Positive long storyboardId,
        @NotNull @Size(min = 1, max = 500) List<@NotNull @Valid Entry> entries) implements ResourceCommand {
    public record Entry(@NotNull CanvasValueType resourceType, @Positive long resourceId,
                        @NotNull @PositiveOrZero Integer expectedVersion, @NotNull ObjectNode patch) {
    }

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.STORYBOARD;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.BATCH_PATCH_STORYBOARD;
    }
}
