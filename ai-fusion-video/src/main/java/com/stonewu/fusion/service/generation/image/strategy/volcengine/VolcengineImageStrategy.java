package com.stonewu.fusion.service.generation.image.strategy.volcengine;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.volcengine.ark.runtime.model.images.generation.GenerateImagesRequest;
import com.volcengine.ark.runtime.model.images.generation.ImagesResponse;
import com.volcengine.ark.runtime.service.ArkService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.List;

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
    public List<String> generate(String prompt, String modelCode, int width, int height, int count,
            List<String> imageUrls, ApiConfig apiConfig) {
        ArkService.Builder builder = ArkService.builder()
                .apiKey(apiConfig.getApiKey())
                .connectTimeout(Duration.ofMinutes(1))
                .timeout(Duration.ofMinutes(25));
        if (StrUtil.isNotBlank(apiConfig.getApiUrl())) {
            builder.baseUrl(apiConfig.getApiUrl());
        }
        ArkService service = builder.build();

        try {
            GenerateImagesRequest.Builder reqBuilder = GenerateImagesRequest.builder()
                    .model(modelCode)
                    .prompt(prompt)
                    .watermark(false)
                    .size(width + "x" + height);

            // 图生图：传入参考图片 URL 列表
            if (imageUrls != null && !imageUrls.isEmpty()) {
                reqBuilder.image(imageUrls);
            }

            GenerateImagesRequest request = reqBuilder.build();

            log.info("[Volcengine] 调用文生图 API: model={}, prompt={}, size={}x{}", modelCode, prompt, width, height);

            ImagesResponse response = service.generateImages(request);
            List<ImagesResponse.Image> images = response.getData();

            if (images == null || images.isEmpty()) {
                throw new RuntimeException("火山引擎返回空结果");
            }

            return images.stream().map(ImagesResponse.Image::getUrl).toList();
        } finally {
            service.shutdownExecutor();
        }
    }

    @Override
    public String submit(ImageTask task, ApiConfig apiConfig) {
        AiModel model = resolveModel(task);
        String modelCode = (model != null && StrUtil.isNotBlank(model.getCode())) ? model.getCode()
                : "doubao-seedream-3-0-t2i-250415";
        int[] size = resolveDefaultSize(model, task);
        int count = (task.getCount() != null && task.getCount() > 0) ? task.getCount() : 1;

        // 解析参考图（图生图场景）
        List<String> imageUrls = parseRefImageUrls(task.getRefImageUrls());

        // 复用纯 API 调用
        List<String> urls = generate(task.getPrompt(), modelCode, size[0], size[1], count, imageUrls, apiConfig);

        // 更新数据库记录
        List<ImageItem> items = imageGenerationService.listItems(task.getId());
        for (int i = 0; i < urls.size() && i < items.size(); i++) {
            ImageItem item = items.get(i);
            item.setImageUrl(urls.get(i));
            item.setStatus(1);
            imageGenerationService.updateItem(item);
        }

        task.setSuccessCount(Math.min(urls.size(), items.size()));
        imageGenerationService.update(task);

        log.info("[Volcengine] 文生图完成: taskId={}, imageCount={}", task.getTaskId(), urls.size());
        return task.getTaskId();
    }

    @Override
    public void poll(String platformTaskId, ImageTask task, ApiConfig apiConfig) {
        // 火山引擎 createImages 是同步 API，submit 中已处理完成，无需轮询
    }

    private AiModel resolveModel(ImageTask task) {
        if (task.getModelId() != null) {
            AiModel model = aiModelService.getById(task.getModelId());
            if (model != null && StrUtil.isNotBlank(model.getCode())) {
                return model;
            }
        }
        return null;
    }
}
