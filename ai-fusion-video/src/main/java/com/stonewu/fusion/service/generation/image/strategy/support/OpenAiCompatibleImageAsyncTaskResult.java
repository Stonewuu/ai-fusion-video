package com.stonewu.fusion.service.generation.image.strategy.support;

import java.util.List;

/** Normalized result of an asynchronous image task query. */
public record OpenAiCompatibleImageAsyncTaskResult(
        String status,
        boolean failed,
        boolean completed,
        List<String> urls,
        String errorMessage
) {
    public OpenAiCompatibleImageAsyncTaskResult {
        urls = urls == null ? List.of() : List.copyOf(urls);
    }
}
