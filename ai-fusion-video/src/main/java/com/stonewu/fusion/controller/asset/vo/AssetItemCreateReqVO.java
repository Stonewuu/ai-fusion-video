package com.stonewu.fusion.controller.asset.vo;

import com.fasterxml.jackson.databind.node.ObjectNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.PositiveOrZero;

public record AssetItemCreateReqVO(@Positive long assetId, @NotNull @PositiveOrZero Integer parentVersion,
                                   @NotBlank String requestId, @NotNull ObjectNode fields) {
}
