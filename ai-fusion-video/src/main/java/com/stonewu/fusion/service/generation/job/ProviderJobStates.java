package com.stonewu.fusion.service.generation.job;

import java.util.List;
import java.util.Locale;
import java.util.function.Supplier;

/**
 * 已登记供应商状态的显式映射；未知响应不伪装成成功或可重新提交的失败。
 */
public final class ProviderJobStates {
    private ProviderJobStates() {
    }

    public static RemoteGenerationResult result(String status, String error, int interval,
                                                Supplier<List<RemoteGenerationResult.Output>> outputs) {
        String normalized = status == null ? "" : status.trim().toLowerCase(Locale.ROOT);
        return switch (normalized) {
            case "succeeded", "success", "completed" -> RemoteGenerationResult.success(outputs.get());
            case "failed", "error", "expired" -> RemoteGenerationResult.failed(error);
            case "cancelled", "canceled" -> RemoteGenerationResult.cancelled(error);
            case "pending", "queued", "running", "in_progress", "processing", "submitted", "waiting", "created" ->
                    RemoteGenerationResult.pending(interval);
            case "unknown", "not_found" -> throw new GenerationJobException(409, "PROVIDER_TASK_UNKNOWN",
                    "供应商暂时无法确认此任务，需要核实受理状态");
            default -> throw new GenerationJobException(502, "PROVIDER_STATUS_INVALID", "供应商返回了未登记的任务状态");
        };
    }
}
