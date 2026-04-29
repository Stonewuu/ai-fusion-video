package com.stonewu.fusion.service.generation.strategy.impl;

import com.alibaba.cloud.ai.dashscope.api.DashScopeImageApi;
import com.alibaba.cloud.ai.dashscope.image.DashScopeImageModel;
import com.alibaba.cloud.ai.dashscope.image.DashScopeImageOptions;
import com.alibaba.cloud.ai.dashscope.spec.DashScopeApiSpec.DashScopeImageAsyncResponse;
import com.alibaba.dashscope.aigc.imagegeneration.ImageGeneration;
import com.alibaba.dashscope.aigc.imagegeneration.ImageGenerationMessage;
import com.alibaba.dashscope.aigc.imagegeneration.ImageGenerationParam;
import com.alibaba.dashscope.aigc.imagegeneration.ImageGenerationResult;
import com.alibaba.dashscope.protocol.Protocol;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.dashscope.DashScopeGenerationSupport;
import com.stonewu.fusion.service.generation.ImageGenerationService;
import com.stonewu.fusion.service.generation.strategy.ImageGenerationStrategy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.image.ImagePrompt;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * 阿里云百炼 DashScope 图片生成策略。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DashScopeImageStrategy implements ImageGenerationStrategy {

    private static final long POLL_INTERVAL_MILLIS = 5000L;
    private static final int DEFAULT_POLL_TIMEOUT_SECONDS = 3600;
    private static final int MAX_POLL_COUNT = (int) ((DEFAULT_POLL_TIMEOUT_SECONDS * 1000L) / POLL_INTERVAL_MILLIS);

    private final ImageGenerationService imageGenerationService;
    private final AiModelService aiModelService;
    private final DashScopeGenerationSupport dashScopeSupport;

    @Override
    public String getName() {
        return DashScopeGenerationSupport.PLATFORM;
    }

    @Override
    public List<String> generate(String prompt, String modelCode, int width, int height, int count,
                                 List<String> imageUrls, ApiConfig apiConfig) {
        DashScopeImageResult result = generateInternal(prompt, modelCode, width + "*" + height, count,
                imageUrls, apiConfig, new JSONObject());
        return result.imageUrls();
    }

    @Override
    public String submit(ImageTask task, ApiConfig apiConfig) {
        AiModel model = resolveModel(task);
        String modelCode = model != null && StrUtil.isNotBlank(model.getCode()) ? model.getCode() : "wan2.7-image";
        JSONObject config = DashScopeGenerationSupport.parseModelConfig(model);
        int[] size = resolveDefaultSize(model, task);
        String sizeText = resolveSizeText(task, config, size[0], size[1]);
        int count = task.getCount() != null && task.getCount() > 0 ? task.getCount() : 1;
        List<String> refImageUrls = parseRefImageUrls(task.getRefImageUrls());

        DashScopeImageResult result = generateInternal(task.getPrompt(), modelCode, sizeText, count, refImageUrls,
                apiConfig, config);
        updateItems(task, result, size[0], size[1]);
        log.info("[DashScope] 图片生成完成: taskId={}, model={}, imageCount={}", task.getTaskId(), modelCode,
                result.imageUrls().size());
        return result.platformTaskId() != null ? result.platformTaskId() : task.getTaskId();
    }

    @Override
    public void poll(String platformTaskId, ImageTask task, ApiConfig apiConfig) {
        // DashScope 图片策略在 submit 中同步等待异步任务完成，消费者无需重复轮询。
    }

    private DashScopeImageResult generateInternal(String prompt, String modelCode, String sizeText, int count,
                                                  List<String> imageUrls, ApiConfig apiConfig, JSONObject config) {
        DashScopeGenerationSupport.requireApiKey(apiConfig);
        String normalizedModelCode = StrUtil.blankToDefault(modelCode, "wan2.7-image").trim();
        int normalizedCount = Math.max(1, count);
        List<String> mediaUrls = prepareReferenceImages(imageUrls);
        boolean legacyPromptOnlyModel = isLegacyPromptOnlyModel(normalizedModelCode);
        if (legacyPromptOnlyModel && !mediaUrls.isEmpty()) {
            throw new BusinessException("DashScope 模型 " + normalizedModelCode + " 不支持参考图输入");
        }
        if (shouldUseSpringAiImageSdk(normalizedModelCode, mediaUrls)) {
            return generateWithSdk(prompt, normalizedModelCode, sizeText, normalizedCount, mediaUrls, apiConfig, config);
        }

        // Spring AI Alibaba 1.1.2.0 尚未暴露 Wan2.6/2.7 与 Qwen-Image 2.0 的 messages 多模态请求对象；
        // DashScope Java SDK 2.22.x 已提供 ImageGeneration 高层 API，可兼容这些模型且无需手写 HTTP。
        return generateWithDashScopeJavaSdk(prompt, normalizedModelCode, sizeText, normalizedCount, mediaUrls,
                apiConfig, config);
    }

    private DashScopeImageResult generateWithSdk(String prompt, String modelCode, String sizeText, int count,
                                                 List<String> imageUrls, ApiConfig apiConfig, JSONObject config) {
        DashScopeImageModel imageModel = buildSdkImageModel(apiConfig, modelCode, sizeText, count, imageUrls, config);
        ImagePrompt imagePrompt = new ImagePrompt(prompt, buildSdkImageOptions(modelCode, sizeText, count, imageUrls, config));
        String taskId = imageModel.submitImageGenTask(imagePrompt);
        if (StrUtil.isBlank(taskId)) {
            throw new BusinessException("DashScope 图片 SDK 未返回 task_id");
        }
        DashScopeImageAsyncResponse finalResponse = waitForSdkImageTask(imageModel, taskId);
        List<String> urls = extractImageUrls(finalResponse);
        if (urls.isEmpty()) {
            throw new BusinessException("DashScope 图片 SDK 任务成功但未返回图片地址: " + finalResponse);
        }
        return new DashScopeImageResult(urls, taskId);
    }

    private DashScopeImageModel buildSdkImageModel(ApiConfig apiConfig, String modelCode, String sizeText, int count,
                                                   List<String> imageUrls, JSONObject config) {
        DashScopeImageApi imageApi = DashScopeImageApi.builder()
                .baseUrl(DashScopeGenerationSupport.resolveRootBaseUrl(apiConfig.getApiUrl()))
                .apiKey(apiConfig.getApiKey())
                .build();
        return DashScopeImageModel.builder()
                .dashScopeApi(imageApi)
                .defaultOptions(buildSdkImageOptions(modelCode, sizeText, count, imageUrls, config))
                .build();
    }

    private DashScopeImageOptions buildSdkImageOptions(String modelCode, String sizeText, int count,
                                                       List<String> imageUrls, JSONObject config) {
        DashScopeImageOptions.Builder builder = DashScopeImageOptions.builder()
                .model(modelCode)
                .n(count);
        if (StrUtil.isNotBlank(sizeText)) {
            builder.withSize(sizeText);
        }
        if (imageUrls != null && imageUrls.size() == 1) {
            builder.refImg(imageUrls.get(0));
        }
        String negativePrompt = firstNonBlank(
                DashScopeGenerationSupport.getString(config, "negativePrompt"),
                DashScopeGenerationSupport.getString(config, "negative_prompt"));
        if (StrUtil.isNotBlank(negativePrompt)) {
            builder.negativePrompt(negativePrompt);
        }
        if (config != null && config.containsKey("promptExtend")) {
            builder.promptExtend(DashScopeGenerationSupport.getBoolean(config, "promptExtend", false));
        } else if (config != null && config.containsKey("prompt_extend")) {
            builder.promptExtend(DashScopeGenerationSupport.getBoolean(config, "prompt_extend", false));
        }
        if (config != null && config.containsKey("watermark")) {
            builder.watermark(DashScopeGenerationSupport.getBoolean(config, "watermark", false));
        }
        Long seed = DashScopeGenerationSupport.getLong(config, "seed");
        if (seed != null && seed >= Integer.MIN_VALUE && seed <= Integer.MAX_VALUE) {
            builder.seed(seed.intValue());
        }
        return builder.build();
    }

    private DashScopeImageAsyncResponse waitForSdkImageTask(DashScopeImageModel imageModel, String taskId) {
        for (int i = 0; i < MAX_POLL_COUNT; i++) {
            DashScopeImageAsyncResponse response = imageModel.getImageGenTask(taskId);
            String status = extractTaskStatus(response);
            if ("SUCCEEDED".equalsIgnoreCase(status)) {
                return response;
            }
            if ("FAILED".equalsIgnoreCase(status) || "CANCELED".equalsIgnoreCase(status)
                    || "UNKNOWN".equalsIgnoreCase(status)) {
                throw new BusinessException("DashScope 图片 SDK 任务失败: status=" + status + ", message="
                        + extractErrorMessage(response));
            }
            sleepQuietly(POLL_INTERVAL_MILLIS);
        }
        throw new BusinessException("DashScope 图片 SDK 任务轮询超时: " + taskId);
    }

    private DashScopeImageResult generateWithDashScopeJavaSdk(String prompt, String modelCode, String sizeText,
                                                              int count, List<String> imageUrls, ApiConfig apiConfig,
                                                              JSONObject config) {
        try {
            ImageGeneration imageGeneration = buildOfficialImageGeneration(apiConfig);
            ImageGenerationParam param = buildOfficialImageParam(prompt, modelCode, sizeText, count,
                    imageUrls, apiConfig, config);
            if (isOfficialAsyncImageModel(modelCode)) {
                ImageGenerationResult imageResult = imageGeneration.asyncCall(param);
                String taskId = extractTaskId(imageResult);
                if (StrUtil.isBlank(taskId)) {
                    throw new BusinessException("DashScope Java SDK 图片任务未返回 task_id: " + imageResult);
                }
                ImageGenerationResult finalResult = waitForOfficialImageTask(imageGeneration, apiConfig, taskId);
                List<String> urls = extractImageUrls(finalResult);
                if (urls.isEmpty()) {
                    throw new BusinessException("DashScope Java SDK 图片任务成功但未返回图片地址: " + finalResult);
                }
                return new DashScopeImageResult(urls, taskId);
            }

            ImageGenerationResult result = imageGeneration.call(param);
            List<String> urls = extractImageUrls(result);
            if (urls.isEmpty()) {
                throw new BusinessException("DashScope Java SDK 图片接口未返回图片地址: " + result);
            }
            return new DashScopeImageResult(urls, null);
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            throw new BusinessException("DashScope Java SDK 图片接口调用异常: " + e.getMessage());
        }
    }

    private ImageGeneration buildOfficialImageGeneration(ApiConfig apiConfig) {
        return new ImageGeneration(Protocol.HTTP.getValue(), DashScopeGenerationSupport.resolveGenerationBaseUrl(apiConfig));
    }

    private ImageGenerationParam buildOfficialImageParam(String prompt, String modelCode, String sizeText,
                                                         int count, List<String> imageUrls, ApiConfig apiConfig,
                                                         JSONObject config) {
        ImageGenerationParam.ImageGenerationParamBuilder<?, ?> builder = ImageGenerationParam.builder();
        builder.apiKey(apiConfig.getApiKey());
        builder.model(modelCode);
        builder.messages(List.of(buildUserMessage(prompt, imageUrls)));
        if (StrUtil.isNotBlank(sizeText)) {
            builder.size(sizeText);
        }
        if (count > 0) {
            builder.n(count);
        }
        String negativePrompt = firstNonBlank(
                DashScopeGenerationSupport.getString(config, "negativePrompt"),
                DashScopeGenerationSupport.getString(config, "negative_prompt"));
        if (StrUtil.isNotBlank(negativePrompt)) {
            builder.negativePrompt(negativePrompt);
        }
        if (config != null && config.containsKey("promptExtend")) {
            builder.promptExtend(DashScopeGenerationSupport.getBoolean(config, "promptExtend", false));
        } else if (config != null && config.containsKey("prompt_extend")) {
            builder.promptExtend(DashScopeGenerationSupport.getBoolean(config, "prompt_extend", false));
        }
        if (config != null && config.containsKey("watermark")) {
            builder.watermark(DashScopeGenerationSupport.getBoolean(config, "watermark", false));
        }
        Long seed = DashScopeGenerationSupport.getLong(config, "seed");
        if (seed != null && seed >= Integer.MIN_VALUE && seed <= Integer.MAX_VALUE) {
            builder.seed(seed.intValue());
        }
        return builder.build();
    }

    private ImageGenerationMessage buildUserMessage(String prompt, List<String> imageUrls) {
        List<Map<String, Object>> content = new ArrayList<>();
        Map<String, Object> text = new LinkedHashMap<>();
        text.put("text", StrUtil.blankToDefault(prompt, ""));
        content.add(text);
        for (String imageUrl : imageUrls) {
            Map<String, Object> image = new LinkedHashMap<>();
            image.put("image", imageUrl);
            content.add(image);
        }
        return ImageGenerationMessage.builder()
                .role("user")
                .content(content)
                .build();
    }

    private ImageGenerationResult waitForOfficialImageTask(ImageGeneration imageGeneration, ApiConfig apiConfig,
                                                           String taskId) throws Exception {
        for (int i = 0; i < MAX_POLL_COUNT; i++) {
            ImageGenerationResult response = imageGeneration.fetch(taskId, apiConfig.getApiKey());
            String status = extractTaskStatus(response);
            if ("SUCCEEDED".equalsIgnoreCase(status)) {
                return response;
            }
            if ("FAILED".equalsIgnoreCase(status) || "CANCELED".equalsIgnoreCase(status)
                    || "UNKNOWN".equalsIgnoreCase(status)) {
                throw new BusinessException("DashScope Java SDK 图片任务失败: status=" + status + ", message="
                        + extractErrorMessage(response));
            }
            sleepQuietly(POLL_INTERVAL_MILLIS);
        }
        throw new BusinessException("DashScope Java SDK 图片任务轮询超时: " + taskId);
    }

    private List<String> extractImageUrls(ImageGenerationResult response) {
        if (response == null || response.getOutput() == null || response.getOutput().getChoices() == null) {
            return List.of();
        }
        List<String> urls = new ArrayList<>();
        response.getOutput().getChoices().forEach(choice -> {
            ImageGenerationMessage message = choice.getMessage();
            if (message == null || message.getContent() == null) {
                return;
            }
            for (Map<String, Object> item : message.getContent()) {
                addIfObjectString(urls, item.get("image"));
                addIfObjectString(urls, item.get("url"));
            }
        });
        return urls.stream().filter(StrUtil::isNotBlank).distinct().toList();
    }

    private List<String> extractImageUrls(DashScopeImageAsyncResponse response) {
        if (response == null || response.output() == null || response.output().results() == null) {
            return List.of();
        }
        return response.output().results().stream()
                .map(DashScopeImageAsyncResponse.DashScopeImageAsyncResponseResult::url)
                .filter(StrUtil::isNotBlank)
                .map(String::trim)
                .distinct()
                .toList();
    }

    private String extractTaskId(ImageGenerationResult response) {
        return response != null && response.getOutput() != null ? response.getOutput().getTaskId() : null;
    }

    private String extractTaskStatus(ImageGenerationResult response) {
        return response != null && response.getOutput() != null ? response.getOutput().getTaskStatus() : null;
    }

    private String extractTaskStatus(DashScopeImageAsyncResponse response) {
        return response != null && response.output() != null ? response.output().taskStatus() : null;
    }

    private String extractErrorMessage(ImageGenerationResult response) {
        if (response == null) {
            return "";
        }
        String code = response.getCode();
        String message = response.getMessage();
        if (StrUtil.isNotBlank(code) && StrUtil.isNotBlank(message)) {
            return code + ": " + message;
        }
        return StrUtil.blankToDefault(message, response.toString());
    }

    private String extractErrorMessage(DashScopeImageAsyncResponse response) {
        if (response == null || response.output() == null) {
            return "";
        }
        String code = response.output().code();
        String message = response.output().message();
        if (StrUtil.isNotBlank(code) && StrUtil.isNotBlank(message)) {
            return code + ": " + message;
        }
        return StrUtil.blankToDefault(message, response.toString());
    }

    private boolean isOfficialAsyncImageModel(String modelCode) {
        String code = modelCode.toLowerCase(Locale.ROOT);
        return code.startsWith("wan2.7-image") || code.startsWith("wan2.6-image");
    }

    private boolean isLegacyPromptOnlyModel(String modelCode) {
        String code = modelCode.toLowerCase(Locale.ROOT);
        return code.startsWith("qwen-image") && !code.startsWith("qwen-image-2.0")
                || code.startsWith("wanx") && code.contains("t2i")
                || code.startsWith("wan2.") && code.contains("t2i");
    }

    private boolean shouldUseSpringAiImageSdk(String modelCode, List<String> imageUrls) {
        String code = modelCode.toLowerCase(Locale.ROOT);
        if (imageUrls != null && imageUrls.size() > 1) {
            return false;
        }
        return !code.startsWith("wan2.7-image")
                && !code.startsWith("wan2.6-image")
                && !code.startsWith("qwen-image-2.0");
    }

    private List<String> prepareReferenceImages(List<String> imageUrls) {
        if (imageUrls == null || imageUrls.isEmpty()) {
            return List.of();
        }
        return imageUrls.stream()
                .filter(StrUtil::isNotBlank)
                .map(url -> {
                    try {
                        return dashScopeSupport.toMediaUrl(url);
                    } catch (IOException e) {
                        throw new BusinessException("DashScope 参考图处理失败: " + e.getMessage());
                    }
                })
                .toList();
    }

    private String resolveSizeText(ImageTask task, JSONObject config, int width, int height) {
        if (task != null && StrUtil.isNotBlank(task.getResolution())) {
            String resolution = task.getResolution().trim();
            if (resolution.matches("\\d+[xX*]\\d+")) {
                return resolution.replace('x', '*').replace('X', '*');
            }
            return resolution;
        }
        String configuredSize = firstNonBlank(
                DashScopeGenerationSupport.getString(config, "defaultSize"),
                DashScopeGenerationSupport.getString(config, "size"));
        if (StrUtil.isNotBlank(configuredSize)) {
            return configuredSize;
        }
        return width + "*" + height;
    }

    private void updateItems(ImageTask task, DashScopeImageResult result, int width, int height) {
        List<ImageItem> items = imageGenerationService.listItems(task.getId());
        List<String> urls = result.imageUrls();
        int successCount = 0;
        for (int i = 0; i < items.size(); i++) {
            ImageItem item = items.get(i);
            item.setPlatformTaskId(result.platformTaskId());
            if (i < urls.size()) {
                item.setImageUrl(urls.get(i));
                item.setWidth(width);
                item.setHeight(height);
                item.setStatus(1);
                successCount++;
            } else {
                item.setStatus(2);
                item.setErrorMsg("DashScope 返回图片数量少于任务数量");
            }
            imageGenerationService.updateItem(item);
        }
        task.setSuccessCount(successCount);
        imageGenerationService.update(task);
    }

    private AiModel resolveModel(ImageTask task) {
        if (task != null && task.getModelId() != null) {
            AiModel model = aiModelService.getById(task.getModelId());
            if (model != null && StrUtil.isNotBlank(model.getCode())) {
                return model;
            }
        }
        return null;
    }

    private void addIfNotBlank(List<String> urls, String url) {
        if (StrUtil.isNotBlank(url)) {
            urls.add(url.trim());
        }
    }

    private void addIfObjectString(List<String> urls, Object value) {
        if (value instanceof String url) {
            addIfNotBlank(urls, url);
        }
    }

    private String firstNonBlank(String... values) {
        if (values == null) {
            return null;
        }
        for (String value : values) {
            if (StrUtil.isNotBlank(value)) {
                return value;
            }
        }
        return null;
    }

    private void sleepQuietly(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BusinessException("DashScope 图片任务轮询被中断");
        }
    }

    private record DashScopeImageResult(List<String> imageUrls, String platformTaskId) {
    }
}
