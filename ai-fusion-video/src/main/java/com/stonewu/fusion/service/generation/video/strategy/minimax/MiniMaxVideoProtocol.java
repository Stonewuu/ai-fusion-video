package com.stonewu.fusion.service.generation.video.strategy.minimax;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.generation.job.ProviderJobStates;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.transport.OfficialVideoContent;

import java.util.List;

/** MiniMax V2 与显式兼容渠道共用的请求内容和任务响应合同。 */
public final class MiniMaxVideoProtocol {
    private MiniMaxVideoProtocol() {
    }

    public static JSONObject request(VideoTask task, AiModel model) {
        var body = new JSONObject().set("model", model.getCode()).set("content", OfficialVideoContent.of(task))
                .set("resolution", task.getResolution()).set("duration", task.getDuration());
        if (StrUtil.isNotBlank(task.getRatio())) {
            body.set("ratio", task.getRatio());
        }
        return body;
    }

    public static RemoteGenerationResult result(JSONObject response) {
        var task = response.getJSONObject("task");
        if (task == null) {
            throw new BusinessException("视频查询结果缺少任务信息");
        }
        var error = task.getJSONObject("error");
        String reason = error == null ? null : error.getStr("code", "") + ": " + error.getStr("message", "");
        return ProviderJobStates.result(task.getStr("status"), reason, 10, () -> {
            var content = task.getJSONObject("content");
            if (content == null || StrUtil.isBlank(content.getStr("url"))) {
                throw new BusinessException("已完成的视频任务没有返回视频地址");
            }
            return List.of(new RemoteGenerationResult.Output(content.getStr("url"), null, null, null,
                    task.getInt("duration")));
        });
    }
}
