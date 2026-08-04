package com.stonewu.fusion.service.generation.image.strategy.comfyui;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiExecutionContext;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiGenerationExecutor;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiPreparedSubmission;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiStoredOutput;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiJobResult;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/** Image generation through an administrator-published ComfyUI workflow. */
@Component
@RequiredArgsConstructor
@Slf4j
public class ComfyUiImageStrategy implements ImageGenerationStrategy {

    private static final long DEFAULT_POLL_INTERVAL_MILLIS = 2_000L;
    private static final long DEFAULT_TIMEOUT_MILLIS = 25L * 60L * 1_000L;

    private final AiModelService aiModelService;
    private final ImageGenerationService imageGenerationService;
    private final ComfyUiGenerationExecutor executor;

    @Override
    public String getName() {
        return ComfyUiWorkflowService.PLATFORM;
    }

    @Override
    public List<String> generate(String prompt,
                                 String modelCode,
                                 int width,
                                 int height,
                                 int count,
                                 List<String> imageUrls,
                                 ApiConfig apiConfig) {
        AiModel model = aiModelService.getByCodeAndApiConfig(
                modelCode, apiConfig != null ? apiConfig.getId() : null);
        if (model == null) {
            throw new BusinessException(404, "找不到对应的 ComfyUI 图片模型");
        }
        return generate(prompt, model, width, height, count, imageUrls, apiConfig);
    }

    @Override
    public List<String> generate(String prompt,
                                 AiModel model,
                                 int width,
                                 int height,
                                 int count,
                                 List<String> imageUrls,
                                 ApiConfig apiConfig) {
        ComfyUiExecutionContext context = executor.resolveActiveContext(model);
        Map<String, Object> values = imageValues(
                prompt, width, height, count, imageUrls, randomSeed());
        ComfyUiPreparedSubmission submission = executor.prepare(
                context, "direct-image", values);
        executor.submit(submission);
        ComfyUiJobResult job = executor.waitForJob(
                context, submission.promptId(), pollInterval(model), timeout(model));
        return executor.storeOutputs(context, job).stream()
                .filter(output -> "image".equals(output.mediaType())
                        && "primary".equals(output.role()))
                .map(ComfyUiStoredOutput::url)
                .limit(Math.max(1, count))
                .toList();
    }

    @Override
    public String submit(ImageTask task, ApiConfig apiConfig) {
        AiModel model = requireModel(task);
        ComfyUiExecutionContext context = executor.resolveContext(model, task.getWorkflowVersionId());
        int[] size = resolveDefaultSize(model, task);
        List<String> references = parseRefImageUrls(task.getRefImageUrls());
        ComfyUiPreparedSubmission submission = executor.prepare(
                context,
                task.getTaskId(),
                imageValues(task.getPrompt(), size[0], size[1], task.getCount(), references, randomSeed()));
        List<ImageItem> items = imageGenerationService.listItems(task.getId());
        if (items.isEmpty()) {
            throw new BusinessException(400, "图片任务缺少生成条目");
        }
        for (ImageItem item : items) {
            item.setPlatformTaskId(submission.promptId());
            imageGenerationService.updateItem(item);
        }
        executor.submit(submission);
        log.info("[ComfyUI Image] 已提交: taskId={}, promptId={}, workflowVersionId={}",
                task.getTaskId(), submission.promptId(), task.getWorkflowVersionId());
        return submission.promptId();
    }

    @Override
    public void poll(String platformTaskId, ImageTask task, ApiConfig apiConfig) {
        AiModel model = requireModel(task);
        ComfyUiExecutionContext context = executor.resolveContext(model, task.getWorkflowVersionId());
        ComfyUiJobResult job = executor.waitForJob(
                context, platformTaskId, pollInterval(model), timeout(model));
        List<ComfyUiStoredOutput> images = executor.storeOutputs(context, job).stream()
                .filter(output -> "image".equals(output.mediaType())
                        && "primary".equals(output.role()))
                .toList();
        List<ImageItem> items = imageGenerationService.listItems(task.getId());
        if (images.size() < items.size()) {
            throw new BusinessException(502,
                    "ComfyUI 返回图片数量不足，期望 " + items.size() + "，实际 " + images.size());
        }
        for (int index = 0; index < items.size(); index++) {
            ImageItem item = items.get(index);
            ComfyUiStoredOutput output = images.get(index);
            item.setPlatformTaskId(platformTaskId);
            item.setImageUrl(output.url());
            item.setFileSize(output.size());
            item.setStatus(1);
            item.setErrorMsg(null);
            imageGenerationService.updateItem(item);
        }
        task.setSuccessCount(items.size());
        imageGenerationService.update(task);
    }

    @Override
    public boolean cancel(String platformTaskId, ImageTask task, ApiConfig apiConfig) {
        return executor.cancel(
                executor.resolveContext(requireModel(task), task.getWorkflowVersionId()),
                platformTaskId);
    }

    @Override
    public boolean persistsResults() {
        return true;
    }

    private AiModel requireModel(ImageTask task) {
        AiModel model = task == null || task.getModelId() == null
                ? null : aiModelService.getById(task.getModelId());
        if (model == null) throw new BusinessException(404, "ComfyUI 图片模型不存在");
        return model;
    }

    private Map<String, Object> imageValues(String prompt,
                                            int width,
                                            int height,
                                            int count,
                                            List<String> references,
                                            long seed) {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("prompt", StrUtil.blankToDefault(prompt, ""));
        values.put("width", width);
        values.put("height", height);
        values.put("count", Math.max(1, count));
        values.put("seed", seed);
        if (references != null && !references.isEmpty()) {
            values.put("referenceImages", new ArrayList<>(references));
        }
        return values;
    }

    private long pollInterval(AiModel model) {
        return configLong(model, "comfyuiPollIntervalMillis", DEFAULT_POLL_INTERVAL_MILLIS);
    }

    private long timeout(AiModel model) {
        return configLong(model, "comfyuiTimeoutMillis", DEFAULT_TIMEOUT_MILLIS);
    }

    private long configLong(AiModel model, String key, long fallback) {
        if (model == null || StrUtil.isBlank(model.getConfig())) return fallback;
        try {
            JSONObject config = JSONUtil.parseObj(model.getConfig());
            Long value = config.getLong(key);
            return value != null && value > 0 ? value : fallback;
        } catch (RuntimeException ignored) {
            return fallback;
        }
    }

    private long randomSeed() {
        return ThreadLocalRandom.current().nextLong(Long.MAX_VALUE);
    }
}
