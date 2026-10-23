package com.stonewu.fusion.controller.storyboard.vo;

import com.fasterxml.jackson.databind.node.ObjectNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.PositiveOrZero;

public record StoryboardPatchReqVO(@Positive long id, @NotNull @PositiveOrZero Integer version,
                                   @NotBlank String requestId, @NotNull ObjectNode patch) {
}
