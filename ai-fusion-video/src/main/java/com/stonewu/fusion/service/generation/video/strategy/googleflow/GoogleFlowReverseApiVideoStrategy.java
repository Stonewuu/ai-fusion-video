package com.stonewu.fusion.service.generation.video.strategy.googleflow;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.googleflow.GoogleFlowReverseApiClient;
import com.stonewu.fusion.service.ai.googleflow.GoogleFlowReverseApiSupport;
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
    public String submit(VideoTask task) {
        validateUnsupportedInputs(task);

        AiModel model = resolveModel(task);
        ApiConfig apiConfig = resolveApiConfig(model);
        GoogleFlowReverseApiSupport.ResolvedVideoRequest request = GoogleFlowReverseApiSupport.resolveVideoRequest(model, task);
        List<String> imageUrls = buildInputImages(task);
        List<VideoItem> items = videoGenerationService.listItems(task.getId());
        String firstRequestId = null;
        int successCount = 0;

        for (VideoItem item : items) {
            GoogleFlowReverseApiClient.CompletionResult completion = client.generate(
                    task.getPrompt(),
                    request.actualModelCode(),
                    imageUrls,
                    apiConfig
            );
            String videoUrl = GoogleFlowReverseApiSupport.extractVideoUrl(completion.content());
            if (StrUtil.isBlank(videoUrl)) {
                throw new BusinessException("GoogleFlowReverseApi 未返回视频地址");
            }

            if (firstRequestId == null && StrUtil.isNotBlank(completion.requestId())) {
                firstRequestId = completion.requestId();
            }

            item.setPlatformTaskId(StrUtil.blankToDefault(completion.requestId(), task.getTaskId()));
            item.setVideoUrl(videoUrl);
            item.setDuration(task.getDuration());
            item.setStatus(1);
            videoGenerationService.updateItem(item);
            successCount++;
        }

        task.setSuccessCount(successCount);
        videoGenerationService.update(task);

        log.info("[GoogleFlowReverseApi][Video] 任务完成: taskId={}, model={}, aspectRatio={}, resolution={}, count={}, imageCount={}",
                task.getTaskId(), request.actualModelCode(), request.aspectRatio(), request.resolution(), successCount, request.imageCount());
        return StrUtil.blankToDefault(firstRequestId, task.getTaskId());
    }

    @Override
    public void poll(String platformTaskId, VideoTask task) {
        // Flow2API 在 submit 阶段已同步返回最终结果
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

    private AiModel resolveModel(VideoTask task) {
        if (task.getModelId() == null) {
            throw new BusinessException("GoogleFlowReverseApi 视频任务缺少 modelId");
        }
        AiModel model = aiModelService.getById(task.getModelId());
        if (model == null || StrUtil.isBlank(model.getCode())) {
            throw new BusinessException("GoogleFlowReverseApi 视频模型不存在或未配置 code");
        }
        return model;
    }

    private ApiConfig resolveApiConfig(AiModel model) {
        if (model.getApiConfigId() == null) {
            throw new BusinessException("GoogleFlowReverseApi 视频模型缺少 apiConfigId");
        }
        ApiConfig apiConfig = apiConfigService.getById(model.getApiConfigId());
        if (apiConfig == null) {
            throw new BusinessException("GoogleFlowReverseApi API 配置不存在");
        }
        return apiConfig;
    }
}
