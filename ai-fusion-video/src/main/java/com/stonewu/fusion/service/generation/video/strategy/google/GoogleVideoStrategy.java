package com.stonewu.fusion.service.generation.video.strategy.google;

import cn.hutool.core.bean.BeanUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.crypto.digest.DigestUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.google.GoogleNativeApi;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import com.stonewu.fusion.service.media.GenerationMediaService;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Set;

@Component
@RequiredArgsConstructor
public class GoogleVideoStrategy implements VideoGenerationStrategy {
    private final GoogleNativeApi google;
    private final GenerationModelCapabilityService capabilities;
    private final GenerationMediaService media;

    @Override
    public String getName() {
        return "gemini";
    }

    @Override
    public Set<String> getSupportedProtocols() {
        return Set.of("gemini", "vertex_ai");
    }

    @Override
    public PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig api) {
        var config = capabilities.getMergedModelConfig(model);
        var target = target(api, config);
        String endpoint = GoogleVideoProtocol.omni(model) ? google.interactionsUrl(target)
                : google.modelUrl(target, model.getCode(), "predictLongRunning");
        var body = GoogleVideoProtocol.request(task, model, config, google.developerApi(target));
        var request = google.post(target, endpoint, body);
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
            var response = google.execute(target, request);
            String id = response.getStr(GoogleVideoProtocol.omni(model) ? "id" : "name");
            if (StrUtil.isBlank(id)) {
                throw new BusinessException("Google 未返回视频任务编号，请查看任务详情");
            }
            return GenerationProviderReceipt.accepted(id);
        });
    }

    @Override
    public RemoteGenerationResult query(String id, VideoTask task, AiModel model, ApiConfig api) {
        var target = target(api, capabilities.getMergedModelConfig(model));
        boolean omni = GoogleVideoProtocol.omni(model);
        JSONObject response;
        if (omni) {
            response = google.getJson(target, google.interactionUrl(target, id));
        } else if (google.developerApi(target)) {
            if ((!id.startsWith("models/" + model.getCode() + "/operations/") && !id.startsWith("operations/"))
                    || !id.matches("[A-Za-z0-9._/-]+") || id.contains("..")) {
                throw new BusinessException("Google 返回的视频任务编号无效");
            }
            response = google.getJson(target, google.developerBase(target) + "/" + id);
        } else {
            response = google.postJson(target, google.modelUrl(target, model.getCode(), "fetchPredictOperation"),
                    new JSONObject().set("operationName", id));
        }
        var result = GoogleVideoProtocol.result(response, omni, google.developerApi(target), task.getDuration());
        if (result.state() != RemoteGenerationResult.State.SUCCEEDED) {
            return result;
        }
        // 长视频不将 Base64 写进数据库；只读查询可重试，媒体接收按任务与输出槽位保持幂等。
        var outputs = new ArrayList<RemoteGenerationResult.Output>();
        for (int index = 0; index < result.outputs().size(); index++) {
            var output = result.outputs().get(index);
            String slot = "google-" + DigestUtil.sha256Hex(id).substring(0, 24) + "-" + index;
            String reference = media.video(task, slot, output.media(), target, MediaKind.VIDEO);
            outputs.add(RemoteGenerationResult.Output.video(reference, null, output.durationSeconds()));
        }
        return RemoteGenerationResult.success(outputs);
    }

    private ApiConfig target(ApiConfig api, JSONObject config) {
        if (google.developerApi(api) || !config.containsKey("googleLocation")) {
            return api;
        }
        ApiConfig target = BeanUtil.copyProperties(api, ApiConfig.class);
        target.setApiUrl(config.getStr("googleLocation"));
        return target;
    }
}
