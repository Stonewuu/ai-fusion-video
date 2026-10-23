package com.stonewu.fusion.controller.storyboard.vo;

import com.fasterxml.jackson.databind.node.ObjectNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.PositiveOrZero;

public record StoryboardChildCreateReqVO(@Positive long parentId, @NotNull @PositiveOrZero Integer parentVersion,
                                         @NotBlank String requestId, @NotNull ObjectNode fields) {
}
