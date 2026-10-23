package com.stonewu.fusion.service.generation.image.strategy.googleflow;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.googleflow.GoogleFlowReverseApiClient;
import com.stonewu.fusion.service.ai.googleflow.GoogleFlowReverseApiSupport;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Google Flow Reverse API 图片生成策略。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class GoogleFlowReverseApiImageStrategy implements ImageGenerationStrategy {

    private final ImageGenerationService imageGenerationService;
    private final AiModelService aiModelService;
    private final GoogleFlowReverseApiClient client;

    @Override
    public String getName() {
        return "google_flow";
    }


    @Override
    public PreparedGenerationCall prepare(ImageTask task, AiModel model, ApiConfig apiConfig) {
        var request = GoogleFlowReverseApiSupport.resolveImageRequest(model, task);
        var call = client.prepare(task.getPrompt(), request.actualModelCode(),
                parseRefImageUrls(task.getRefImageUrls()), apiConfig);
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
            var response = call.get();
            String image = GoogleFlowReverseApiSupport.extractImageUrl(response.content());
            if (StrUtil.isBlank(image)) {
                throw new BusinessException("Google Flow 未返回图片地址");
            }
            return new GenerationProviderReceipt(response.requestId(),
                    RemoteGenerationResult.success(List.of(RemoteGenerationResult.Output.image(image))));
        });
    }

    @Override
    public RemoteGenerationResult query(String platformTaskId, ImageTask task, AiModel model, ApiConfig apiConfig) {
        throw new IllegalStateException("同步图片结果必须从已保存回执读取");
    }


}
