package com.stonewu.fusion.service.generation.image.strategy.dashscope;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import com.alibaba.cloud.ai.dashscope.api.DashScopeImageApi;
import com.alibaba.cloud.ai.dashscope.image.DashScopeImageModel;
import com.alibaba.cloud.ai.dashscope.image.DashScopeImageOptions;
import com.alibaba.cloud.ai.dashscope.spec.DashScopeApiSpec.DashScopeImageAsyncResponse;
import com.alibaba.dashscope.aigc.imagegeneration.ImageGenerationMessage;
import com.alibaba.dashscope.aigc.imagegeneration.ImageGenerationParam;
import com.alibaba.dashscope.aigc.imagegeneration.ImageGenerationResult;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.dashscope.DashScopeGenerationSupport;
import com.stonewu.fusion.service.ai.dashscope.DashScopeSingleRequestClient;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.generation.job.*;
import com.stonewu.fusion.service.generation.transport.GenerationHttpClients;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.image.ImagePrompt;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.*;

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
    public int batchSize(ImageTask task, AiModel model, ApiConfig apiConfig) {
        return task.getCount();
    }

    @Override
    public PreparedGenerationCall prepare(ImageTask task, AiModel model, ApiConfig apiConfig) {
        DashScopeGenerationSupport.requireApiKey(apiConfig);
        String code = model.getCode();
        JSONObject config = DashScopeGenerationSupport.parseModelConfig(model);
        int[] size = resolveDefaultSize(model, task);
        String sizeText = resolveSizeText(task, config, size[0], size[1]);
        var references = prepareReferenceImages(parseRefImageUrls(task.getRefImageUrls()));
        if (isLegacyPromptOnlyModel(code) && !references.isEmpty()) {
            throw new BusinessException("当前 DashScope 模型不支持参考图");
        }
        if (shouldUseSpringAiImageSdk(code, references)) {
            var client = buildSdkImageModel(apiConfig, code, sizeText, task.getCount(), references, config);
            var prompt = new ImagePrompt(task.getPrompt(),
                    buildSdkImageOptions(code, sizeText, task.getCount(), references, config));
            return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () ->
                    GenerationProviderReceipt.accepted(client.submitImageGenTask(prompt)));
        }
        var client = buildOfficialImageGeneration(apiConfig);
        var param = buildOfficialImageParam(task.getPrompt(), code, sizeText, task.getCount(), references, apiConfig,
                config);
        boolean async = isOfficialAsyncImageModel(code);
        var call = client.prepareImage(param, async);
        return new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
            var response = call.get();
            if (async) {
                return GenerationProviderReceipt.accepted(extractTaskId(response));
            }
            return GenerationProviderReceipt.completed(
                    RemoteGenerationResult.success(extractImageUrls(response).stream()
                            .map(RemoteGenerationResult.Output::image).toList()));
        });
    }

    @Override
    public RemoteGenerationResult query(String platformTaskId, ImageTask task, AiModel model, ApiConfig apiConfig) {
        DashScopeGenerationSupport.requireApiKey(apiConfig);
        // 查询只需要原输入数量选择协议；素材失效不能阻止观察已经发出的远端任务。
        var references = parseRefImageUrls(task.getRefImageUrls());
        if (shouldUseSpringAiImageSdk(model.getCode(), references)) {
            var config = DashScopeGenerationSupport.parseModelConfig(model);
            int[] size = resolveDefaultSize(model, task);
            var client = buildSdkImageModel(apiConfig, model.getCode(), resolveSizeText(task, config, size[0], size[1]),
                    1, references, config);
            var result = client.getImageGenTask(platformTaskId);
            return ProviderJobStates.result(extractTaskStatus(result), extractErrorMessage(result), 5,
                    () -> extractImageUrls(result).stream().map(RemoteGenerationResult.Output::image).toList());
        }
        try {
            var result = ImageGenerationResult.fromDashScopeResult(
                    buildOfficialImageGeneration(apiConfig).fetch(platformTaskId, apiConfig.getApiKey()));
            return ProviderJobStates.result(extractTaskStatus(result), extractErrorMessage(result), 5,
                    () -> extractImageUrls(result).stream().map(RemoteGenerationResult.Output::image).toList());
        } catch (BusinessException failure) {
            throw failure;
        } catch (Exception failure) {
            throw new BusinessException("DashScope 图片查询请求异常: " + failure.getMessage());
        }
    }


    private DashScopeImageModel buildSdkImageModel(ApiConfig apiConfig, String modelCode, String sizeText, int count,
                                                   List<String> imageUrls, JSONObject config) {
        DashScopeImageApi imageApi = DashScopeImageApi.builder()
                .baseUrl(DashScopeGenerationSupport.resolveRootBaseUrl(apiConfig.getApiUrl()))
                .apiKey(apiConfig.getApiKey())
                .restClientBuilder(GenerationHttpClients.restClientBuilder(apiConfig))
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


    private DashScopeSingleRequestClient buildOfficialImageGeneration(ApiConfig apiConfig) {
        return new DashScopeSingleRequestClient(apiConfig);
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


    private record DashScopeImageResult(List<String> imageUrls, String platformTaskId) {
    }
}
