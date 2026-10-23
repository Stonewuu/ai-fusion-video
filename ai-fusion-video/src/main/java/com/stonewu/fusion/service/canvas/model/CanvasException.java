package com.stonewu.fusion.service.canvas.model;

import com.stonewu.fusion.common.BusinessException;
import lombok.Getter;

import java.util.Map;

@Getter
public class CanvasException extends BusinessException {
    private final CanvasErrorCode errorCode;
    private final Map<String, Object> details;

    public CanvasException(CanvasErrorCode errorCode, String message) {
        this(400, errorCode, message, Map.of());
    }

    public CanvasException(int status, CanvasErrorCode errorCode,
                           String message, Map<String, Object> details) {
        super(status, message);
        this.errorCode = errorCode;
        this.details = Map.copyOf(details);
    }
}
