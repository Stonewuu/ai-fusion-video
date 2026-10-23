package com.stonewu.fusion.service.resource.model;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

import java.util.List;

/**
 * 分集版本包含场次变更；批量命令以该版本保护整个有序场次集合。
 */
public record ScriptScenesCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @Positive long resourceId,
        @NotNull @PositiveOrZero Integer expectedVersion,
        @NotNull Mode mode,
        @NotNull @Size(max = 500) List<@NotNull ObjectNode> scenes,
        @NotNull @Size(max = 5000) List<@NotNull @Positive Long> sceneIds) implements ResourceCommand {
    public enum Mode {APPEND, REPLACE, DELETE, REORDER}

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.SCRIPT_EPISODE;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.MANAGE_SCRIPT_SCENES;
    }
}
