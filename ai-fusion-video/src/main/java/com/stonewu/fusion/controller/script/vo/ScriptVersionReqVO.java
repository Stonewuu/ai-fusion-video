package com.stonewu.fusion.controller.script.vo;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PositiveOrZero;

public record ScriptVersionReqVO(@NotNull @PositiveOrZero Integer version, @NotBlank String requestId) {
}
