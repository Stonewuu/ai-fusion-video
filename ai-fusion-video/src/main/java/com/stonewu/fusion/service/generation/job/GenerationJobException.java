package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.common.BusinessException;
import lombok.Getter;

/**
 * 生成受理、调度和恢复的稳定错误码。
 */
@Getter
public class GenerationJobException extends BusinessException {
    private final String errorCode;

    public GenerationJobException(int status, String code, String message) {
        super(status, message);
        this.errorCode = code;
    }

    public static GenerationJobException fenced() {
        return new GenerationJobException(409, "GENERATION_LEASE_LOST", "生成执行权已变化，请读取最新任务状态");
    }

    public static GenerationJobException invalid(String message) {
        return new GenerationJobException(400, "INVALID_GENERATION_REQUEST", message);
    }
}
