package com.stonewu.fusion.service.canvas.run;

import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;

import java.util.Map;

public final class CanvasRunException extends CanvasException {
    public CanvasRunException(int status, String code, String message) {
        super(status, CanvasErrorCode.INVALID_COMMAND, message, Map.of("runError", code));
    }

    public static CanvasRunException invalid(String message) {
        return new CanvasRunException(400, "INVALID_RUN", message);
    }

    public static CanvasRunException conflict(String message) {
        return new CanvasRunException(409, "RUN_CONFLICT", message);
    }

    public static CanvasRunException fenced() {
        return new CanvasRunException(409, "RUN_FENCED", "运行已取消、超时或由其他工作者接管");
    }
}
