package com.stonewu.fusion.controller.asset.vo;

import com.fasterxml.jackson.databind.node.ObjectNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;

public record AssetCreateReqVO(@Positive Long projectId, @NotBlank String requestId, @NotNull ObjectNode fields) {
}
