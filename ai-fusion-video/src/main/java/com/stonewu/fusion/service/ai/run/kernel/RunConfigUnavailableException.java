package com.stonewu.fusion.service.ai.run.kernel;

import com.stonewu.fusion.enums.ai.AgentRuntimeErrorCode;

public final class RunConfigUnavailableException extends RuntimeException {

    private static final AgentRuntimeErrorCode ERROR_CODE =
            AgentRuntimeErrorCode.RUN_CONFIG_UNAVAILABLE;

    public RunConfigUnavailableException(String message) {
        super(message);
    }

    public AgentRuntimeErrorCode errorCode() {
        return ERROR_CODE;
    }

    public String getErrorCode() {
        return ERROR_CODE.getCode();
    }
}
