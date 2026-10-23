package com.stonewu.fusion.service.resource.model;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

import java.util.List;

/**
 * 使用父集合版本保护整批追加、重建、清空、删除或排序，空间布局不参与这里的顺序。
 */
public record StoryboardChildrenCommand(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @NotNull CanvasValueType resourceType,
        @Positive long resourceId,
        @NotNull @PositiveOrZero Integer expectedVersion,
        @NotNull Mode mode,
        @NotNull @Size(max = 500) List<@NotNull ObjectNode> children,
        @NotNull @Size(max = 5000) List<@NotNull @Positive Long> childIds) implements ResourceCommand {
    public enum Mode {APPEND, REPLACE, CLEAR, DELETE, REORDER}

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.MANAGE_STORYBOARD_CHILDREN;
    }
}
