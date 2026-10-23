package com.stonewu.fusion.service.resource.model;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

import java.util.List;

public record AssetBatchCreateCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @NotNull CanvasValueType resourceType, @Positive Long projectId, @Positive Long assetId,
        @PositiveOrZero Integer expectedParentVersion, @NotNull Boolean reuseExisting,
        @NotNull @Size(min = 1, max = 10) List<@NotNull ObjectNode> entries) implements ResourceCommand {
    @Override
    public ResourceOperation operation() {
        return ResourceOperation.BATCH_CREATE_ASSETS;
    }
}
