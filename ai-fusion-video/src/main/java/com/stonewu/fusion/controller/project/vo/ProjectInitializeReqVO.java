package com.stonewu.fusion.controller.project.vo;

import jakarta.validation.constraints.*;

public record ProjectInitializeReqVO(
        @NotBlank @Size(max = 128) @Pattern(regexp = "[A-Za-z0-9._:-]+") String requestId,
        @NotNull @PositiveOrZero Integer version) {
}
