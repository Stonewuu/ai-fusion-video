package com.stonewu.fusion.controller.ai.vo;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

import java.util.List;

public record ProviderSetupReqVO(
        @Valid @NotNull ApiConfigSaveReqVO configuration,
        @NotBlank @Pattern(regexp = "[a-fA-F0-9-]{36}") String requestId,
        @NotEmpty @Size(max = 250) List<@NotBlank String> modelIds) {
}
