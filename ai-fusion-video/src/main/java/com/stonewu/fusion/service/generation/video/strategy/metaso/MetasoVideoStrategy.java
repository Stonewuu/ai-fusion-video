package com.stonewu.fusion.service.generation.video.strategy.metaso;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.transport.OfficialVideoHttpClient;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import com.stonewu.fusion.service.generation.video.strategy.minimax.MiniMaxVideoProtocol;
import org.springframework.stereotype.Component;

import java.util.Map;

/** 秘塔公开视频 API：H3 与 Seedance 共用任务协议，各自能力由平台预设声明。 */
@Component
public final class MetasoVideoStrategy implements VideoGenerationStrategy {
    @Override
    public String getName() {
        return "metaso_video";
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
        var options = JSONUtil.parseObj(model.getConfig());
        boolean contextIr = booleanOption(options, "contextIrEnabled");
        boolean fast = booleanOption(options, "fastMode");
        if (contextIr && !options.getBool("supportContextIr", false)) {
            throw new BusinessException("当前秘塔模型不支持 Context IR");
        }
        if (fast && !options.getBool("supportPriorityChannel", false)) {
            throw new BusinessException("当前秘塔模型不支持优先通道");
        }
        // 显式值优先于平台账号的全局开关，防止同一配置随控制台设置而改变费用和行为。
        body.set("context_ir_enabled", contextIr);
        if (options.getBool("supportPriorityChannel", false)) {
            body.set("fast_mode", fast);
        }
        if (task.getWatermark() != null) {
            body.set("aigc_watermark", task.getWatermark());
        }
        return body;
    }

    private boolean booleanOption(JSONObject options, String name) {
        Object value = options.get(name);
        if (value == null) {
            return false;
        }
        if (!(value instanceof Boolean flag)) {
            throw new BusinessException("秘塔模型参数 " + name + " 必须为开关值");
        }
        return flag;
    }
}
