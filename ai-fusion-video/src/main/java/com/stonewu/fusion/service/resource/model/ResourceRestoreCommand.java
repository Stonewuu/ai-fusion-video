package com.stonewu.fusion.service.resource.model;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record ResourceRestoreCommand(@NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
                                     @NotNull CanvasValueType resourceType, @NotBlank String commandId,
                                     @NotBlank @Pattern(regexp = "[a-f0-9]{64}") String expectedDigest) implements ResourceCommand {
    @Override
    public ResourceOperation operation() {
        return ResourceOperation.RESTORE;
    }
}
