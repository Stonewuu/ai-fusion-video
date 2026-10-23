package com.stonewu.fusion.service.generation.video.strategy.volcengine;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.job.*;
import com.stonewu.fusion.service.generation.transport.OfficialVideoContent;
import com.stonewu.fusion.service.generation.transport.OfficialVideoHttpClient;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * 火山方舟官方 Content Generation V3，使用冻结能力发送实际支持的参数。
 */
@Component
public final class VolcengineVideoStrategy implements VideoGenerationStrategy {
    private static final String TASKS = "/contents/generations/tasks";

    @Override
    public String getName() {
        return "volcengine";
    }

    @Override
    public PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig config) {
        var client = new OfficialVideoHttpClient(config);
        var request = client.post(OfficialVideoHttpClient.baseUrl(config, "/api/v3") + TASKS, buildRequest(task, model),
                Map.of());
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () ->
                GenerationProviderReceipt.accepted(client.execute(request).getStr("id")));
    }

    @Override
    public RemoteGenerationResult query(String taskId, VideoTask task, AiModel model, ApiConfig config) {
        var result = new OfficialVideoHttpClient(config).get(OfficialVideoHttpClient.taskUrl(
                OfficialVideoHttpClient.baseUrl(config, "/api/v3"), TASKS, taskId));
        var error = result.getJSONObject("error");
        String reason = error == null ? null : error.getStr("code", "") + ": " + error.getStr("message", "");
        return ProviderJobStates.result(result.getStr("status"), reason, 10, () -> {
            var content = result.getJSONObject("content");
            if (content == null || StrUtil.isBlank(content.getStr("video_url"))) {
                throw new BusinessException("火山方舟已完成任务没有返回视频地址");
            }
            return List.of(new RemoteGenerationResult.Output(content.getStr("video_url"), null, null,
                    content.getStr("last_frame_url"), result.getInt("duration")));
        });
    }

    public JSONObject buildRequest(VideoTask task, AiModel model) {
        var config = StrUtil.isBlank(model.getConfig()) ? new JSONObject() : JSONUtil.parseObj(model.getConfig());
        var request = new JSONObject().set("model", model.getCode()).set("content", OfficialVideoContent.of(task));
        if (StrUtil.isNotBlank(task.getResolution())) {
            request.set("resolution", task.getResolution());
        }
        if (StrUtil.isNotBlank(task.getRatio())) {
            request.set("ratio", task.getRatio());
        }
        if (task.getDuration() != null) {
            request.set("duration", task.getDuration());
        }
        if (task.getSeed() != null) {
            request.set("seed", task.getSeed());
        }
        if (task.getWatermark() != null) {
            request.set("watermark", task.getWatermark());
        }
        if (task.getGenerateAudio() != null && config.getBool("supportGenerateAudio", true)) {
            request.set("generate_audio", task.getGenerateAudio());
        }
        if (task.getCameraFixed() != null && config.getBool("supportCameraFixed", true)) {
            request.set("camera_fixed", task.getCameraFixed());
        }
        if (config.getBool("supportReturnLastFrame", true)) {
            request.set("return_last_frame", true);
        }
        if (config.containsKey("outputFormat")) {
            request.set("output_format", config.getStr("outputFormat"));
        }
        if (config.containsKey("referenceTaskType")) {
            request.set("omni_reference_task_type", config.getStr("referenceTaskType"));
        }
        return request;
    }
}
