package com.stonewu.fusion.controller.storyboard.vo;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PositiveOrZero;

public record StoryboardVersionReqVO(@NotNull @PositiveOrZero Integer version, @NotBlank String requestId) {
}
