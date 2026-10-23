package com.stonewu.fusion.service.media.model;

import com.stonewu.fusion.service.generation.GenerationKind;

import java.util.Objects;

public record MediaGenerationOrigin(GenerationKind kind, long taskId) {
    public MediaGenerationOrigin {
        Objects.requireNonNull(kind);
        if (taskId <= 0) {
            throw MediaException.invalid("生成来源任务标识无效");
        }
    }
}
