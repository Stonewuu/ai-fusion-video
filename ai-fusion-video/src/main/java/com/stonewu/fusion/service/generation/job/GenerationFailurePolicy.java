package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.media.model.MediaException;

/** 可确定的权限、配置和数据错误需要用户处理，不通过等待重试解决。 */
final class GenerationFailurePolicy {
    private GenerationFailurePolicy() {
    }

    static boolean permanent(Exception failure, String code) {
        if (failure instanceof MediaException && code.equals("MEDIA_INGEST_IN_PROGRESS")) {
            return false;
        }
        if (code.equals("PROVIDER_TASK_UNKNOWN") || code.equals("GENERATION_OUTPUT_COUNT_MISMATCH")) {
            return true;
        }
        if (failure instanceof BusinessException business) {
            int status = business.getCode();
            return status >= 400 && status < 500 && status != 408 && status != 429;
        }
        return false;
    }
}
