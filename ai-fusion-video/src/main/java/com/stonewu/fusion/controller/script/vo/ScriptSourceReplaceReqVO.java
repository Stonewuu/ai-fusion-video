package com.stonewu.fusion.controller.script.vo;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PositiveOrZero;

public record ScriptSourceReplaceReqVO(@NotBlank String requestId, @NotNull @PositiveOrZero Integer version,
                                       @NotBlank String rawContent) {
}
