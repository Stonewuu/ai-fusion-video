package com.stonewu.fusion.service.media.model;

import com.stonewu.fusion.common.BusinessException;

public class MediaException extends BusinessException {
    private final String errorCode;

    public MediaException(int status, String errorCode, String message) {
        super(status, message);
        this.errorCode = errorCode;
    }

    public String getErrorCode() {
        return errorCode;
    }

    public static MediaException invalid(String message) {
        return new MediaException(400, "INVALID_MEDIA", message);
    }

    public static MediaException unavailable(String message) {
        return new MediaException(503, "MEDIA_UNAVAILABLE", message);
    }
}
