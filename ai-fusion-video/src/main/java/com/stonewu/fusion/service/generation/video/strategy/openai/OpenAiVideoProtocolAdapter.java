package com.stonewu.fusion.service.generation.video.strategy.openai;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolAdapter;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolContext;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoTaskResult;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import okhttp3.RequestBody;

/**
 * OpenAI 兼容通用视频协议适配器，默认按官方 Sora Videos API 处理。
 */
@Component
@RequiredArgsConstructor
public class OpenAiVideoProtocolAdapter implements OpenAiCompatibleVideoProtocolAdapter {

    private final OpenAiCompatibleVideoProtocolSupport support;

    @Override
    public String getProtocol() {
        return "openai";
    }

    @Override
    public String resolveSubmitUrl(OpenAiCompatibleVideoProtocolContext context) {
        return support.resolveOpenAiVideosUrl(context.apiConfig());
    }

    @Override
    public RequestBody buildSubmitBody(OpenAiCompatibleVideoProtocolContext context) {
        return support.buildSoraSubmitBody(context);
    }

    @Override
    public OpenAiCompatibleVideoTaskResult parseSubmitResponse(OpenAiCompatibleVideoProtocolContext context,
                                                               String responseBody) {
        return parseResult(responseBody);
    }

    @Override
    public String resolveQueryUrl(OpenAiCompatibleVideoProtocolContext context, String trackingId) {
        return support.resolveOpenAiVideosUrl(context.apiConfig()) + "/" + trackingId;
    }

    @Override
    public OpenAiCompatibleVideoTaskResult parseQueryResponse(OpenAiCompatibleVideoProtocolContext context,
                                                              String responseBody) {
        return parseResult(responseBody);
    }

    @Override
    public String resolveVideoContentUrl(OpenAiCompatibleVideoProtocolContext context,
                                         String trackingId,
                                         OpenAiCompatibleVideoTaskResult result) {
        return support.resolveOpenAiVideosUrl(context.apiConfig()) + "/" + trackingId + "/content";
    }

    @Override
    public String resolveCoverContentUrl(OpenAiCompatibleVideoProtocolContext context,
                                         String trackingId,
                                         OpenAiCompatibleVideoTaskResult result) {
        return support.resolveOpenAiVideosUrl(context.apiConfig()) + "/" + trackingId + "/content?variant=thumbnail";
    }

    private OpenAiCompatibleVideoTaskResult parseResult(String responseBody) {
        JsonNode root = support.readJson(responseBody, "OpenAI 视频响应不是合法 JSON");
        String trackingId = support.firstText(root, "id");
        String status = support.firstText(root, "status");
        Integer duration = support.parsePositiveSeconds(support.firstText(root, "seconds"));
        String errorMessage = support.extractErrorMessage(root);
        return new OpenAiCompatibleVideoTaskResult(trackingId, status, duration, null, null, errorMessage);
    }
}
