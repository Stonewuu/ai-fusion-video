package com.stonewu.fusion.service.generation.image.strategy.volcengine;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.generation.job.GenerationPreparedRequest;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.generation.transport.ArkGenerationClients;
import com.volcengine.ark.runtime.model.images.generation.GenerateImagesRequest;
import com.volcengine.ark.runtime.model.images.generation.ImagesResponse;
import com.volcengine.ark.runtime.service.ArkService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 火山引擎（方舟）图片生成策略
 * <p>
 * 支持 Seedream 系列模型（5.0/4.0/3.0 等）文生图。
 * 通过 volcengine-java-sdk-ark-runtime 的 ArkService 调用。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class VolcengineImageStrategy implements ImageGenerationStrategy {

    private final ImageGenerationService imageGenerationService;
    private final AiModelService aiModelService;

    @Override
    public String getName() {
        return "volcengine";
    }

    @Override
    public PreparedGenerationCall prepare(ImageTask task, AiModel model, ApiConfig apiConfig) {
        int[] size = resolveDefaultSize(model, task);
        var images = parseRefImageUrls(task.getRefImageUrls());
        var builder = GenerateImagesRequest.builder().model(model.getCode()).prompt(task.getPrompt())
                .watermark(false).size(size[0] + "x" + size[1]);
        if (images != null && !images.isEmpty()) {
            builder.image(images);
        }
        var request = builder.build();
        ArkService service = ArkGenerationClients.create(apiConfig);
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
            var result = service.generateImages(request).getData();
            if (result == null || result.isEmpty()) {
                throw new IllegalStateException("火山引擎返回空结果");
            }
            var outputs = result.stream().map(ImagesResponse.Image::getUrl).map(RemoteGenerationResult.Output::image)
                    .toList();
            return GenerationProviderReceipt.completed(RemoteGenerationResult.success(outputs));
        }, service::shutdownExecutor);
    }

    @Override
    public RemoteGenerationResult query(String platformTaskId, ImageTask task, AiModel model, ApiConfig apiConfig) {
        throw new IllegalStateException("同步图片结果必须从已保存回执读取");
    }


}
