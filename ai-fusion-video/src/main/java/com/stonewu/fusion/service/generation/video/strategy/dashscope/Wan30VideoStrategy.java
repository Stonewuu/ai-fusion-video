package com.stonewu.fusion.service.generation.video.strategy.dashscope;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.job.*;
import com.stonewu.fusion.service.generation.transport.OfficialVideoHttpClient;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * Wan 3.0 官方统一 media 协议，独立于旧版 Wan 的媒体字段与 SDK 响应结构。
 */
@Component
public final class Wan30VideoStrategy implements VideoGenerationStrategy {
    @Override
    public String getName() {
        return "dashscope_wan3";
    }

    @Override
    public PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig config) {
        var client = new OfficialVideoHttpClient(config);
        var request = client.post(
                OfficialVideoHttpClient.baseUrl(config, "/api/v1") + "/services/aigc/video-generation/video-synthesis",
                buildRequest(task, model), Map.of("X-DashScope-Async", "enable"));
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
            var response = client.execute(request).getJSONObject("output");
            if (response == null) {
                throw new BusinessException("万相返回的受理结果缺少任务信息");
            }
            return GenerationProviderReceipt.accepted(response.getStr("task_id"));
        });
    }

    @Override
    public RemoteGenerationResult query(String taskId, VideoTask task, AiModel model, ApiConfig config) {
        var result = new OfficialVideoHttpClient(config).get(OfficialVideoHttpClient.taskUrl(
                OfficialVideoHttpClient.baseUrl(config, "/api/v1"), "/tasks", taskId));
        var output = result.getJSONObject("output");
        if (output == null) {
            throw new BusinessException("万相返回的查询结果缺少任务信息");
        }
        String error = output.getStr("code", "") + ": " + output.getStr("message", "");
        return ProviderJobStates.result(output.getStr("task_status"), error, 15, () -> {
            if (StrUtil.isBlank(output.getStr("video_url"))) {
                throw new BusinessException("万相已完成任务没有返回视频地址");
            }
            var usage = result.getJSONObject("usage");
            // Wan 3.0 的 duration 包含输入视频；只能把 output_video_duration 当作成片时长。
            Double duration = usage == null ? null : usage.getDouble("output_video_duration");
            return List.of(new RemoteGenerationResult.Output(output.getStr("video_url"), null, null, null,
                    duration == null ? null : (int) Math.ceil(duration)));
        });
    }

    public static JSONObject buildRequest(VideoTask task, AiModel model) {
        var input = new JSONObject();
        if (StrUtil.isNotBlank(task.getPrompt())) {
            input.set("prompt", task.getPrompt());
        }
        var media = new JSONArray();
        add(media, "first_frame", task.getFirstFrameImageUrl());
        add(media, "last_frame", task.getLastFrameImageUrl());
        addList(media, "reference_image", task.getReferenceImageUrls());
        addList(media, "reference_video", task.getReferenceVideoUrls());
        addList(media, "reference_audio", task.getReferenceAudioUrls());
        if (!media.isEmpty()) {
            input.set("media", media);
        }
        var parameters = new JSONObject().set("resolution", task.getResolution())
                .set("ratio", task.getRatio()).set("duration", task.getDuration());
        if (task.getGenerateAudio() != null) {
            parameters.set("audio", task.getGenerateAudio());
        }
        if (task.getWatermark() != null) {
            parameters.set("watermark", task.getWatermark());
        }
        if (task.getSeed() != null) {
            parameters.set("seed", task.getSeed());
        }
        var config = JSONUtil.parseObj(model.getConfig());
        if (config.containsKey("promptExtend")) {
            parameters.set("prompt_extend", config.getBool("promptExtend"));
        }
        return new JSONObject().set("model", model.getCode()).set("input", input).set("parameters", parameters);
    }

    private static void add(JSONArray media, String type, String url) {
        if (StrUtil.isNotBlank(url)) {
            media.add(new JSONObject().set("type", type).set("url", url));
        }
    }

    private static void addList(JSONArray media, String type, String encoded) {
        if (StrUtil.isBlank(encoded)) {
            return;
        }
        for (String url : JSONUtil.parseArray(encoded).toList(String.class)) {
            add(media, type, url);
        }
    }
}
