package com.stonewu.fusion.service.generation.image.strategy.support;

import okhttp3.RequestBody;

/** A fully materialized HTTP request produced by an image protocol adapter. */
public record OpenAiCompatibleImageRequest(
        String url,
        RequestBody body
) {
}
