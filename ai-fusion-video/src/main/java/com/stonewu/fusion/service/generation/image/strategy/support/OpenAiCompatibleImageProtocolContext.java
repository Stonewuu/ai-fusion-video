package com.stonewu.fusion.service.generation.image.strategy.support;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;

import java.util.List;
import java.util.Objects;

/**
 * A complete input snapshot for one OpenAI-compatible image request.
 *
 * <p>The protocol adapters deliberately receive this context instead of a
 * growing list of arguments.  This keeps request-shape decisions in the
 * adapter layer and leaves the strategy responsible for orchestration only.</p>
 */
public record OpenAiCompatibleImageProtocolContext(
        AiModel model,
        ApiConfig apiConfig,
        String modelCode,
        String prompt,
        int width,
        int height,
        int count,
        List<String> imageUrls,
        JSONObject modelConfig,
        boolean asyncMode
) {

    public OpenAiCompatibleImageProtocolContext {
        imageUrls = imageUrls == null ? List.of() : imageUrls.stream()
                .filter(Objects::nonNull)
                .toList();
    }

    public boolean hasReferenceImages() {
        return !imageUrls.isEmpty();
    }

    public OpenAiCompatibleImageProtocolContext withCount(int requestCount) {
        return new OpenAiCompatibleImageProtocolContext(
                model, apiConfig, modelCode, prompt, width, height, requestCount,
                imageUrls, modelConfig, asyncMode);
    }
}
