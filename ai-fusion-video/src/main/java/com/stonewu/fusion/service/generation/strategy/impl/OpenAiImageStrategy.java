package com.stonewu.fusion.service.generation.strategy.impl;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.common.BusinessException;
import com.openai.client.OpenAIClient;
import com.openai.client.okhttp.OpenAIOkHttpClient;
import com.openai.models.images.Image;
import com.openai.models.images.ImageGenerateParams;
import com.openai.models.images.ImageModel;
import com.openai.models.images.ImagesResponse;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import com.stonewu.fusion.service.generation.ImageGenerationService;
import com.stonewu.fusion.service.generation.strategy.ImageGenerationStrategy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * OpenAI 图片生成策略
 * <p>
 * 支持 DALL·E 3、DALL·E 2、gpt-image-1 等模型文生图。
 * 通过 openai-java SDK 的 client.images().generate() 调用。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class OpenAiImageStrategy implements ImageGenerationStrategy {

    private final ImageGenerationService imageGenerationService;
    private final AiModelService aiModelService;

    @Override
    public String getName() {
        return "openai";
    }

    @Override
    public List<String> generate(String prompt, String modelCode, int width, int height, int count,
                                  List<String> imageUrls, ApiConfig apiConfig) {
        if (imageUrls != null && !imageUrls.isEmpty()) {
            throw new BusinessException("当前图片模型 " + modelCode + " 使用的是 OpenAI 文生图接口，不支持参考图输入");
        }

        OpenAIOkHttpClient.Builder builder = OpenAIOkHttpClient.builder().apiKey(apiConfig.getApiKey());
        if (StrUtil.isNotBlank(apiConfig.getApiUrl())) {
            builder.baseUrl(apiConfig.getApiUrl());
        }
        java.net.Proxy proxy = AiProxySupport.javaProxyOrNull(apiConfig);
        if (proxy != null) {
            builder.proxy(proxy);
        }
        OpenAIClient client = builder.build();

        ImageGenerateParams.Builder paramsBuilder = ImageGenerateParams.builder()
                .prompt(prompt)
                .model(ImageModel.of(modelCode))
                .n((long) count);

        // 尺寸映射
        ImageGenerateParams.Size size = mapSize(width, height);
        if (size != null) {
            paramsBuilder.size(size);
        }

        ImageGenerateParams params = paramsBuilder.build();
        log.info("[OpenAI] 调用文生图 API: model={}, prompt={}, size={}x{}", modelCode, prompt, width, height);

        ImagesResponse response = client.images().generate(params);
        List<Image> images = response.data().orElseThrow(() -> new RuntimeException("OpenAI 返回空结果"));
        if (images.isEmpty()) {
            throw new RuntimeException("OpenAI 返回空图片列表");
        }

        return images.stream()
                .map(img -> img.url().orElseThrow(() -> new RuntimeException("OpenAI 图片未返回 URL")))
                .toList();
    }

    @Override
    public String submit(ImageTask task, ApiConfig apiConfig) {
        AiModel model = resolveModel(task);
        String modelCode = (model != null && StrUtil.isNotBlank(model.getCode())) ? model.getCode() : "dall-e-3";
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

        log.info("[OpenAI] 文生图完成: taskId={}, imageCount={}", task.getTaskId(), urls.size());
        return task.getTaskId();
    }

    @Override
    public void poll(String platformTaskId, ImageTask task, ApiConfig apiConfig) {
        // OpenAI images.generate 是同步 API，submit 中已处理完成，无需轮询
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

    /**
     * 将像素尺寸映射到 OpenAI 支持的枚举值
     */
    private ImageGenerateParams.Size mapSize(int width, int height) {
        String sizeStr = width + "x" + height;
        return switch (sizeStr) {
            case "256x256" -> ImageGenerateParams.Size._256X256;
            case "512x512" -> ImageGenerateParams.Size._512X512;
            case "1024x1024" -> ImageGenerateParams.Size._1024X1024;
            case "1024x1792" -> ImageGenerateParams.Size._1024X1792;
            case "1792x1024" -> ImageGenerateParams.Size._1792X1024;
            default -> ImageGenerateParams.Size._1024X1024;
        };
    }
}
