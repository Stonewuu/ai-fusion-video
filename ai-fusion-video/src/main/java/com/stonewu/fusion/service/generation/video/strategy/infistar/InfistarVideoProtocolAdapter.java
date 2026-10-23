package com.stonewu.fusion.service.generation.video.strategy.infistar;

import cn.hutool.core.util.StrUtil;
import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolAdapter;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolContext;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoProtocolSupport;
import com.stonewu.fusion.service.generation.video.strategy.support.OpenAiCompatibleVideoTaskResult;
import lombok.RequiredArgsConstructor;
import okhttp3.HttpUrl;
import okhttp3.RequestBody;
import org.springframework.stereotype.Component;

/** 无限星河公开 Videos 合同；复用 multipart 请求与持久化任务执行。 */
@Component
@RequiredArgsConstructor
public class InfistarVideoProtocolAdapter implements OpenAiCompatibleVideoProtocolAdapter {
    private final OpenAiCompatibleVideoProtocolSupport support;

    @Override
    public String getProtocol() {
        return "infistar_video";
    }

    @Override
    public String resolveSubmitUrl(OpenAiCompatibleVideoProtocolContext context) {
        return support.resolveOpenAiVideosUrl(context.apiConfig());
    }

    @Override
    public RequestBody buildSubmitBody(OpenAiCompatibleVideoProtocolContext context) {
        var task = context.task();
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl())
                || !support.parseJsonUrls(task.getReferenceImageUrls()).isEmpty()
                || !support.parseJsonUrls(task.getReferenceVideoUrls()).isEmpty()
                || !support.parseJsonUrls(task.getReferenceAudioUrls()).isEmpty()) {
            throw new BusinessException("当前无限星河 Videos 接口仅接入单张首帧；尾帧和参考素材参数尚未接入，不能忽略后继续提交");
        }
        boolean platformSize = context.modelConfig().getBool("useProviderDefaultSize", false);
        if (platformSize && (!"平台默认".equals(context.task().getResolution())
                || !"adaptive".equals(context.task().getRatio()))) {
            throw new BusinessException("此模型使用平台默认尺寸，请将分辨率设为平台默认、画面比例设为自适应");
        }
        return support.buildMultipartVideosBody(context, !platformSize);
    }

    @Override
    public OpenAiCompatibleVideoTaskResult parseSubmitResponse(OpenAiCompatibleVideoProtocolContext context,
                                                               String responseBody) {
        return result(responseBody);
    }

    @Override
    public String resolveQueryUrl(OpenAiCompatibleVideoProtocolContext context, String trackingId) {
        return HttpUrl.get(resolveSubmitUrl(context)).newBuilder().addPathSegment(trackingId).build().toString();
    }

    @Override
    public OpenAiCompatibleVideoTaskResult parseQueryResponse(OpenAiCompatibleVideoProtocolContext context,
                                                              String responseBody) {
        return result(responseBody);
    }

    @Override
    public String resolveVideoContentUrl(OpenAiCompatibleVideoProtocolContext context, String trackingId,
                                         OpenAiCompatibleVideoTaskResult result) {
        return resolveQueryUrl(context, trackingId) + "/content";
    }

    private OpenAiCompatibleVideoTaskResult result(String responseBody) {
        JsonNode root = support.readJson(responseBody, "无限星河视频响应不是合法 JSON");
        // 公开指南命名 task_id，OpenAI 兼容 Videos 对象命名 id；仅支持这两个明确的合同字段。
        String taskId = support.firstText(root, "task_id");
        String compatibleId = support.firstText(root, "id");
        if (taskId != null && compatibleId != null && !taskId.equals(compatibleId)) {
            throw new BusinessException("无限星河视频响应包含冲突的任务标识");
        }
        return new OpenAiCompatibleVideoTaskResult(taskId != null ? taskId : compatibleId,
                support.firstText(root, "status"),
                support.parsePositiveSeconds(support.firstText(root, "seconds")),
                null, null, support.extractErrorMessage(root));
    }
}
