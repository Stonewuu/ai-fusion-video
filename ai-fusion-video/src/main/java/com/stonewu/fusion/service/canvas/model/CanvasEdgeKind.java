package com.stonewu.fusion.service.canvas.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Arrays;

public enum CanvasEdgeKind {
    EXECUTION("execution"), REFERENCE("reference"), ANNOTATION("annotation");

    private final String code;

    CanvasEdgeKind(String code) {
        this.code = code;
    }

    @JsonValue
    public String code() {
        return code;
    }

    @JsonCreator
    public static CanvasEdgeKind fromCode(String value) {
        return Arrays.stream(values()).filter(kind -> kind.code.equals(value)).findFirst()
                .orElseThrow(() -> new CanvasException(CanvasErrorCode.INVALID_GRAPH,
                        "不支持的画布关系类型: " + value));
    }
}
