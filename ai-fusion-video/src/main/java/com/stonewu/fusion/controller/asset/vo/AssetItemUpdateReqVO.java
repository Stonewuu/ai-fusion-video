package com.stonewu.fusion.controller.asset.vo;

import com.fasterxml.jackson.databind.node.ObjectNode;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.PositiveOrZero;

public record AssetItemUpdateReqVO(@Positive long id, @NotNull @PositiveOrZero Integer version,
                                   @NotNull @PositiveOrZero Integer assetVersion,
                                   @NotBlank String requestId, @NotNull ObjectNode patch) {
}
