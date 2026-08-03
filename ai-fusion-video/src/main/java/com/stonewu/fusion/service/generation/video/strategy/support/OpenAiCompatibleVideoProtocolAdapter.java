package com.stonewu.fusion.service.generation.video.strategy.support;

import okhttp3.RequestBody;

/**
 * OpenAI 兼容视频协议适配器。
 */
public interface OpenAiCompatibleVideoProtocolAdapter {

    String getProtocol();

    String resolveSubmitUrl(OpenAiCompatibleVideoProtocolContext context);

    RequestBody buildSubmitBody(OpenAiCompatibleVideoProtocolContext context);

    OpenAiCompatibleVideoTaskResult parseSubmitResponse(OpenAiCompatibleVideoProtocolContext context,
                                                        String responseBody);

    String resolveQueryUrl(OpenAiCompatibleVideoProtocolContext context, String trackingId);

    OpenAiCompatibleVideoTaskResult parseQueryResponse(OpenAiCompatibleVideoProtocolContext context,
                                                       String responseBody);

    default String resolveVideoContentUrl(OpenAiCompatibleVideoProtocolContext context,
                                          String trackingId,
                                          OpenAiCompatibleVideoTaskResult result) {
        return null;
    }

    default String resolveCoverContentUrl(OpenAiCompatibleVideoProtocolContext context,
                                          String trackingId,
                                          OpenAiCompatibleVideoTaskResult result) {
        return null;
    }
}
