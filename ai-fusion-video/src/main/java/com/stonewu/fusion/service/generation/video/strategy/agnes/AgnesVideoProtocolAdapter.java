package com.stonewu.fusion.service.generation.video.strategy.agnes;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolAdapter;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolContext;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoTaskResult;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.RequiredArgsConstructor;
import okhttp3.RequestBody;
import org.springframework.stereotype.Component;

/**
 * Agnes 视频协议适配器。
 */
@Component
@RequiredArgsConstructor
public class AgnesVideoProtocolAdapter implements OpenAiCompatibleVideoProtocolAdapter {

    private final OpenAiCompatibleVideoProtocolSupport support;

    @Override
    public String getProtocol() {
        return "agnes";
    }

    @Override
    public String resolveSubmitUrl(OpenAiCompatibleVideoProtocolContext context) {
        return support.resolveFixedV1VideosUrl(context.apiConfig());
    }

    @Override
    public RequestBody buildSubmitBody(OpenAiCompatibleVideoProtocolContext context) {
        return support.buildAgnesSubmitBody(context);
    }

    @Override
    public OpenAiCompatibleVideoTaskResult parseSubmitResponse(OpenAiCompatibleVideoProtocolContext context,
                                                               String responseBody) {
        JsonNode root = support.readJson(responseBody, "Agnes 视频任务提交响应不是合法 JSON");
        String trackingId = firstNonBlank(
                support.firstText(root, "video_id"),
                support.firstText(root, "id"),
                support.firstText(root, "task_id")
        );
        String status = support.firstText(root, "status", "state");
        Integer duration = support.parsePositiveSeconds(support.firstText(root, "seconds", "duration"));
        String errorMessage = support.extractErrorMessage(root);
        return new OpenAiCompatibleVideoTaskResult(trackingId, status, duration, null, null, errorMessage);
    }

    @Override
    public String resolveQueryUrl(OpenAiCompatibleVideoProtocolContext context, String trackingId) {
        return support.resolveAgnesQueryUrl(context, trackingId);
    }

    @Override
    public OpenAiCompatibleVideoTaskResult parseQueryResponse(OpenAiCompatibleVideoProtocolContext context,
                                                              String responseBody) {
        JsonNode root = support.readJson(responseBody, "Agnes 视频任务查询响应不是合法 JSON");
        String trackingId = firstNonBlank(
                support.firstText(root, "video_id"),
                support.firstText(root, "id"),
                support.firstText(root, "task_id")
        );
        String status = support.firstText(root, "status", "state");
        Integer duration = support.parsePositiveSeconds(support.firstText(root, "seconds", "duration"));
        JsonNode metadata = root.path("metadata");
        // Agnes 官方完成态视频地址位于 metadata.url；其余字段兼容旧网关/代理返回。
        String videoUrl = firstNonBlank(
                support.firstText(metadata, "url", "video_url", "videoUrl"),
                support.firstText(root, "video_url", "url", "remixed_from_video_id", "remixedFromVideoId")
        );
        String coverUrl = firstNonBlank(
                support.firstText(metadata, "cover_url", "coverUrl", "thumbnail_url", "thumbnailUrl"),
                support.firstText(root, "cover_url", "coverUrl", "thumbnail_url", "thumbnailUrl")
        );
        String errorMessage = support.extractErrorMessage(root);
        return new OpenAiCompatibleVideoTaskResult(trackingId, status, duration, videoUrl, coverUrl, errorMessage);
    }

    private String firstNonBlank(String... values) {
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                return value;
            }
        }
        return null;
    }
}
