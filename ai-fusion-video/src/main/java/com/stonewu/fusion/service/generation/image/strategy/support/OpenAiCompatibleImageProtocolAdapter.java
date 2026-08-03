package com.stonewu.fusion.service.generation.image.strategy.support;

import java.util.List;

/**
 * Protocol-specific request/response contract for OpenAI-compatible image APIs.
 */
public interface OpenAiCompatibleImageProtocolAdapter {

    String getProtocol();

    OpenAiCompatibleImageRequest buildRequest(OpenAiCompatibleImageProtocolContext context);

    List<String> parseImageUrls(OpenAiCompatibleImageProtocolContext context, String responseBody);

    /** Whether the upstream accepts only one image per generation request. */
    default boolean isSingleImagePerRequest(OpenAiCompatibleImageProtocolContext context) {
        return false;
    }
}
