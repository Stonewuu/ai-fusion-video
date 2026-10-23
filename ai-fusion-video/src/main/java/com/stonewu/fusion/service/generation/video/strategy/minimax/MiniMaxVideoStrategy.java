package com.stonewu.fusion.service.generation.video.strategy.minimax;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.job.*;
import com.stonewu.fusion.service.generation.transport.OfficialVideoHttpClient;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * MiniMax 官方视频 V2；模型能力与参数合法性由统一的视频契约校验。
 */
@Component
public final class MiniMaxVideoStrategy implements VideoGenerationStrategy {
    @Override
    public String getName() {
        return "minimax_h3";
    }

    @Override
    public PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig config) {
        var client = new OfficialVideoHttpClient(config);
        var request = client.post(OfficialVideoHttpClient.baseUrl(config, "/v2") + "/video_generation",
                buildRequest(task, model), Map.of());
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () ->
                GenerationProviderReceipt.accepted(client.execute(request).getStr("task_id")));
    }

    @Override
    public RemoteGenerationResult query(String taskId, VideoTask task, AiModel model, ApiConfig config) {
        var client = new OfficialVideoHttpClient(config);
        return MiniMaxVideoProtocol.result(client.get(OfficialVideoHttpClient.taskUrl(
                OfficialVideoHttpClient.baseUrl(config, "/v2"), "/query/video_generation", taskId)));
    }

    public JSONObject buildRequest(VideoTask task, AiModel model) {
        var body = MiniMaxVideoProtocol.request(task, model);
        var options = StrUtil.isBlank(model.getConfig()) ? new JSONObject() : JSONUtil.parseObj(model.getConfig());
        if (options.containsKey("promptExpansionMode")) {
            body.set("extra", new JSONObject().set("prompt_expansion_mode", options.getStr("promptExpansionMode")));
        }
        return body;
    }

}
