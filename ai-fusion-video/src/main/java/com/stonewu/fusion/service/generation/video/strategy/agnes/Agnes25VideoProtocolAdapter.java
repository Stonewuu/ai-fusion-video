package com.stonewu.fusion.service.generation.video.strategy.agnes;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolAdapter;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolContext;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoTaskResult;
import lombok.RequiredArgsConstructor;
import okhttp3.RequestBody;
import org.springframework.stereotype.Component;

/**
 * 显式的 Agnes 2.5 协议，不将旧版任务编号或 remix 来源当作结果。
 */
@Component
@RequiredArgsConstructor
public class Agnes25VideoProtocolAdapter implements OpenAiCompatibleVideoProtocolAdapter {
    private final OpenAiCompatibleVideoProtocolSupport support;

    @Override
    public String getProtocol() {
        return Agnes25VideoStrategy.PROTOCOL;
    }

    @Override
    public String resolveSubmitUrl(OpenAiCompatibleVideoProtocolContext context) {
        return support.resolveFixedV1VideosUrl(context.apiConfig());
    }

    @Override
    public RequestBody buildSubmitBody(OpenAiCompatibleVideoProtocolContext context) {
        return Agnes25VideoRequest.build(context);
    }

    @Override
    public String resolveQueryUrl(OpenAiCompatibleVideoProtocolContext context, String trackingId) {
        return support.resolveAgnesQueryUrl(context, trackingId);
    }

    @Override
    public OpenAiCompatibleVideoTaskResult parseSubmitResponse(OpenAiCompatibleVideoProtocolContext context,
                                                               String body) {
        return parse(body, false);
    }

    @Override
    public OpenAiCompatibleVideoTaskResult parseQueryResponse(OpenAiCompatibleVideoProtocolContext context,
                                                              String body) {
        return parse(body, true);
    }

    private OpenAiCompatibleVideoTaskResult parse(String body, boolean query) {
        JsonNode root = support.readJson(body, "Agnes 视频响应格式不正确");
        return new OpenAiCompatibleVideoTaskResult(support.firstText(root, "video_id"),
                support.firstText(root, "status"), support.parsePositiveSeconds(support.firstText(root, "seconds")),
                query ? support.firstText(root, "url") : null, null, support.extractErrorMessage(root));
    }
}
