package com.stonewu.fusion.service.media.model;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.constraints.*;

/**
 * 业务字段就是来源身份；不接收 URL、所有者或推测出的媒体类别。
 */
public record MediaSourceReference(@NotNull MediaSourceType type, @Positive long id,
                                   @NotNull @Min(0) Integer version, @NotBlank @Size(max = 96) String field) {
    public MediaSourceReference(CanvasValueType type, long id, Integer version, String field) {
        this(MediaSourceType.business(type), id, version, field);
    }

    @JsonAnySetter
    public void unknown(String name, JsonNode value) {
        throw MediaException.invalid("媒体来源不接受额外字段：" + name);
    }
}
