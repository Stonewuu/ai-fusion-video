package com.stonewu.fusion.service.canvas.model;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * 运行结果或画布固定快照二选一；旧已存图省略 snapshotId 时仍按运行结果解析。
 */
public record CanvasOutputReference(String nodeRunId, String outputId, String resourceId,
                                    @JsonInclude(JsonInclude.Include.NON_NULL) String snapshotId) {
    public CanvasOutputReference(String nodeRunId, String outputId, String resourceId) {
        this(nodeRunId, outputId, resourceId, null);
    }
}
