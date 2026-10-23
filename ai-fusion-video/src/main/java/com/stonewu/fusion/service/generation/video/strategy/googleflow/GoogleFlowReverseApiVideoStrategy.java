package com.stonewu.fusion.service.generation.video.strategy.googleflow;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.googleflow.GoogleFlowReverseApiClient;
import com.stonewu.fusion.service.ai.googleflow.GoogleFlowReverseApiSupport;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import com.stonewu.fusion.service.generation.video.strategy.VideoGenerationStrategy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * Google Flow Reverse API 视频生成策略。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class GoogleFlowReverseApiVideoStrategy implements VideoGenerationStrategy {

    private final AiModelService aiModelService;
    private final ApiConfigService apiConfigService;
    private final VideoGenerationService videoGenerationService;
    private final GoogleFlowReverseApiClient client;

    @Override
    public String getName() {
        return "google_flow";
    }

    @Override
    public PreparedGenerationCall prepare(VideoTask task, AiModel model, ApiConfig apiConfig) {
        validateUnsupportedInputs(task);
        var request = GoogleFlowReverseApiSupport.resolveVideoRequest(model, task);
        var call = client.prepare(task.getPrompt(), request.actualModelCode(), buildInputImages(task), apiConfig);
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
            var response = call.get();
            String video = GoogleFlowReverseApiSupport.extractVideoUrl(response.content());
            if (StrUtil.isBlank(video)) {
                throw new BusinessException("Google Flow 未返回视频地址");
            }
            return new GenerationProviderReceipt(response.requestId(), RemoteGenerationResult.success(List.of(
                    RemoteGenerationResult.Output.video(video, null, task.getDuration()))));
        });
    }

    @Override
    public RemoteGenerationResult query(String platformTaskId, VideoTask task, AiModel model, ApiConfig apiConfig) {
        throw new IllegalStateException("同步视频结果必须从已保存回执读取");
    }

    private void validateUnsupportedInputs(VideoTask task) {
        if (!GoogleFlowReverseApiSupport.parseJsonUrls(task.getReferenceVideoUrls()).isEmpty()) {
            throw new BusinessException("GoogleFlowReverseApi 当前仅支持文本与参考图片输入，暂不支持参考视频");
        }
        if (!GoogleFlowReverseApiSupport.parseJsonUrls(task.getReferenceAudioUrls()).isEmpty()) {
            throw new BusinessException("GoogleFlowReverseApi 当前仅支持文本与参考图片输入，暂不支持参考音频");
        }
    }

    private List<String> buildInputImages(VideoTask task) {
        List<String> imageUrls = new ArrayList<>();
        if (StrUtil.isNotBlank(task.getFirstFrameImageUrl())) {
            imageUrls.add(task.getFirstFrameImageUrl());
        }
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl())) {
            imageUrls.add(task.getLastFrameImageUrl());
        }
        imageUrls.addAll(GoogleFlowReverseApiSupport.parseJsonUrls(task.getReferenceImageUrls()));
        return imageUrls;
    }


}
