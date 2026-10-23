package com.stonewu.fusion.controller.storyboard.vo;

import jakarta.validation.constraints.*;

public record ComposeVideoReqVO(@NotNull @PositiveOrZero Integer version,
                                @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
                                @NotNull Boolean includeBgm) {
}
