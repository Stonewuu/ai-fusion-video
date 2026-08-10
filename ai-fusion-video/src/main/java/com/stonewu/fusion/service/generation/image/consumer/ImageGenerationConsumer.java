package com.stonewu.fusion.service.generation.image.consumer;

import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.infrastructure.queue.RedisTaskQueue;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategyRouter;
import com.stonewu.fusion.service.storage.MediaStorageService;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 生图任务消费器
 * <p>
 * 从 Redis 队列中取出任务，通过对应的策略执行生图
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ImageGenerationConsumer {

    private static final String BASE_QUEUE_NAME = "image_generation";
    private static final String MODEL_QUEUE_PREFIX = BASE_QUEUE_NAME + ":model:";
    private static final int MODEL_TYPE_IMAGE = 2;

    private final RedisTaskQueue taskQueue;
    private final ImageGenerationService imageGenerationService;
    private final AiModelService aiModelService;
    private final ApiConfigService apiConfigService;
    private final GenerationModelCapabilityService generationModelCapabilityService;
    private final ReferenceImageTransportService referenceImageTransportService;
    private final ImageGenerationStrategyRouter imageGenerationStrategyRouter;
    private final MediaStorageService mediaStorageService;
    private final ComfyUiWorkflowService comfyUiWorkflowService;

    private final AtomicInteger workerThreadCounter = new AtomicInteger(1);
    private final ExecutorService workerExecutor = Executors.newCachedThreadPool(r -> {
        Thread thread = new Thread(r, "image-generation-worker-" + workerThreadCounter.getAndIncrement());
        thread.setDaemon(true);
        return thread;
    });

    /**
     * 提交生图任务到队列
     */
    public String submitTask(ImageTask task) {
        AiModel queueModel = resolveQueueModel(task.getModelId());
        if (queueModel == null) {
            throw new BusinessException("没有可用的图片生成模型");
        }
        task.setModelId(queueModel.getId());
        pinWorkflowVersion(queueModel, task);
        generationModelCapabilityService.validateImageTask(queueModel, task);

        String queueName = resolveQueueName(task.getModelId());
        String taskId = IdUtil.fastSimpleUUID();
        task.setTaskId(taskId);
        task.setStatus(0); // PENDING
        imageGenerationService.create(task);
        refreshQueueMaxConcurrent(queueName, task.getModelId());

        // 为每个 count 创建 ImageItem
        for (int i = 0; i < task.getCount(); i++) {
            ImageItem item = ImageItem.builder()
                    .taskId(task.getId())
                    .status(0)
                    .build();
            imageGenerationService.createItem(item);
        }

        // 入队
        taskQueue.push(queueName, taskId);
        log.info("[ImageConsumer] 任务入队: taskId={}, queue={}, modelId={}", taskId, queueName, task.getModelId());
        return taskId;
    }

    /**
     * 提交任务并同步等待结果（阻塞当前线程）
     * <p>
     * 适用于 AI Agent 工具等需要同步获取生图结果的场景。
     * 内部流程：submitTask() 入队 → Consumer 定时取出执行 → 本方法轮询 DB 等待完成。
     *
     * @param task      生图任务（需设置好 prompt、width、height、modelId 等）
     * @param timeoutMs 最大等待时间（毫秒）
     * @return 完成的 ImageTask（含结果图片在 ImageItem 中）
     * @throws RuntimeException 超时或任务失败时抛出
     * @throws InterruptedException 等待过程中线程被中断
     */
    public ImageTask submitAndWait(ImageTask task, long timeoutMs) throws InterruptedException {
        String taskId = submitTask(task);

        long deadline = System.currentTimeMillis() + timeoutMs;
        long pollInterval = 2000L;

        log.info("[ImageConsumer] 同步等待任务完成: taskId={}, timeout={}ms", taskId, timeoutMs);

        while (System.currentTimeMillis() < deadline) {
            Thread.sleep(pollInterval);

            ImageTask current = imageGenerationService.getByTaskId(taskId);
            switch (current.getStatus()) {
                case 2: // 已完成
                    log.info("[ImageConsumer] 任务已完成: taskId={}", taskId);
                    return current;
                case 3: // 失败
                    String errorMsg = current.getErrorMsg() != null ? current.getErrorMsg() : "未知错误";
                    throw new RuntimeException("生图任务失败: " + errorMsg);
                default:
                    // 0-排队中 1-处理中，继续等待
                    break;
            }
        }

        // 超时：标记任务失败，避免 consumer 继续执行无意义的任务
        imageGenerationService.updateStatus(task.getId(), 3, "同步等待超时");
        throw new RuntimeException("生图任务排队超时（等待 " + (timeoutMs / 1000) + " 秒），当前任务较多，请稍后重试");
    }

    public boolean cancelTask(String taskId, Long userId) {
        ImageTask task = imageGenerationService.getByTaskId(taskId);
        if (task == null || userId == null || !userId.equals(task.getUserId())) {
            throw new BusinessException(404, "图片任务不存在");
        }
        if (Integer.valueOf(2).equals(task.getStatus()) || Integer.valueOf(3).equals(task.getStatus())) {
            return false;
        }
        String queueName = resolveQueueName(task.getModelId());
        if (Integer.valueOf(0).equals(task.getStatus()) && taskQueue.remove(queueName, taskId)) {
            markCancelled(task);
            return true;
        }
        AiModel model = aiModelService.getById(task.getModelId());
        ApiConfig apiConfig = model == null || model.getApiConfigId() == null
                ? null : apiConfigService.getById(model.getApiConfigId());
        if (model == null || apiConfig == null) {
            throw new BusinessException("图片任务缺少模型或 API 配置，无法取消");
        }
        ImageGenerationStrategy strategy = imageGenerationStrategyRouter.resolve(model, apiConfig);
        Set<String> platformTaskIds = new LinkedHashSet<>();
        imageGenerationService.listItems(task.getId()).stream()
                .map(ImageItem::getPlatformTaskId)
                .filter(StrUtil::isNotBlank)
                .forEach(platformTaskIds::add);
        boolean cancelled = false;
        for (String platformTaskId : platformTaskIds) {
            cancelled |= strategy.cancel(platformTaskId, task, apiConfig);
        }
        if (!cancelled) {
            throw new BusinessException(400, "当前图片任务尚不能取消或远端任务已结束");
        }
        markCancelled(task);
        return true;
    }

    private void markCancelled(ImageTask task) {
        for (ImageItem item : imageGenerationService.listItems(task.getId())) {
            if (!Integer.valueOf(1).equals(item.getStatus())) {
                item.setStatus(2);
                item.setErrorMsg("用户取消");
                imageGenerationService.updateItem(item);
            }
        }
        imageGenerationService.updateStatus(task.getId(), 3, "用户取消");
    }

    /**
     * 定时消费队列
     */
    @Scheduled(fixedDelay = 3000)
    public void consume() {
        for (String queueName : collectQueueNamesToConsume()) {
            drainQueue(queueName);
        }
    }

    private void drainQueue(String queueName) {
        while (true) {
            String taskId = taskQueue.acquireAndPop(queueName, 1);
            if (taskId == null) {
                return;
            }
            dispatchTask(queueName, taskId);
        }
    }

    private void dispatchTask(String queueName, String taskId) {
        workerExecutor.execute(() -> {
            try {
                taskQueue.markRunning(queueName, taskId, 30);
                processTask(queueName, taskId);
            } catch (Exception e) {
                log.error("[ImageConsumer] 任务处理失败: taskId={}", taskId, e);
            } finally {
                taskQueue.markComplete(queueName, taskId);
                taskQueue.release(queueName);
            }
        });
    }

    private void processTask(String queueName, String taskId) {
        ImageTask task;
        try {
            task = imageGenerationService.getByTaskId(taskId);
        } catch (Exception e) {
            log.error("[ImageConsumer] 任务不存在: taskId={}", taskId);
            return;
        }

        refreshQueueMaxConcurrent(queueName, task.getModelId());

        // 更新状态为处理中
        imageGenerationService.updateStatus(task.getId(), 1, null);

        ImageGenerationStrategy strategy;
        ApiConfig apiConfig = null;
        AiModel model = null;
        if (task.getModelId() != null) {
            try {
                model = aiModelService.getById(task.getModelId());
                if (model != null) {
                    if (model.getApiConfigId() != null) {
                        apiConfig = apiConfigService.getById(model.getApiConfigId());
                    }
                }
            } catch (Exception e) {
                log.warn("[ImageConsumer] 模型配置获取失败: modelId={}", task.getModelId());
            }
        }
        if (model == null || apiConfig == null) {
            imageGenerationService.updateStatus(task.getId(), 3, "找不到匹配的 API 配置");
            log.error("[ImageConsumer] 找不到模型或匹配的 API 配置: modelId={}", task.getModelId());
            return;
        }

        try {
            strategy = imageGenerationStrategyRouter.resolve(model, apiConfig);
            generationModelCapabilityService.validateImageTask(model, task);
            List<String> referenceImages = referenceImageTransportService.parseJsonInputs(
                    task.getRefImageUrls(), "refImageUrls");
            if (!referenceImages.isEmpty()) {
                List<String> resolvedInputs = referenceImageTransportService.resolveInputs(
                        model,
                        generationModelCapabilityService.getMergedModelConfig(model),
                        referenceImages,
                        apiConfig);
                task.setRefImageUrls(JSONUtil.toJsonStr(resolvedInputs));
            }
            String platformTaskId = strategy.submit(task, apiConfig);
            log.info("[ImageConsumer] 任务已提交到平台: taskId={}, platformTaskId={}", taskId, platformTaskId);

            // 轮询结果
            strategy.poll(platformTaskId, task, apiConfig);

            // 持久化远程图片到本地/OSS 存储
            if (!strategy.persistsResults()) {
                persistImageItems(task);
            }

            // 更新为完成
            imageGenerationService.updateStatus(task.getId(), 2, null);
        } catch (Exception e) {
            log.error("[ImageConsumer] 任务执行失败: taskId={}", taskId, e);
            imageGenerationService.updateStatus(task.getId(), 3, e.getMessage());
        }
    }

    private List<String> collectQueueNamesToConsume() {
        List<String> queueNames = new ArrayList<>(taskQueue.listRegisteredQueuesByPrefix(MODEL_QUEUE_PREFIX));
        queueNames.sort(String::compareTo);
        return queueNames;
    }

    private void pinWorkflowVersion(AiModel model, ImageTask task) {
        if (model.getComfyuiWorkflowId() == null) return;
        var workflow = comfyUiWorkflowService.requireWorkflow(model.getComfyuiWorkflowId());
        if (!Integer.valueOf(1).equals(workflow.getStatus())
                || workflow.getActiveVersionId() == null) {
            throw new BusinessException("ComfyUI 工作流已禁用或没有已发布版本");
        }
        var version = comfyUiWorkflowService.requireVersion(workflow.getActiveVersionId());
        task.setWorkflowVersionId(version.getId());
    }

    private void refreshQueueMaxConcurrent(String queueName, Long modelId) {
        int maxConcurrent = resolveQueueMaxConcurrent(modelId);
        taskQueue.setMaxConcurrent(queueName, maxConcurrent);
    }

    private int resolveQueueMaxConcurrent(Long modelId) {
        AiModel model = resolveQueueModel(modelId);
        Integer configured = model != null ? model.getMaxConcurrency() : null;
        return configured != null && configured > 0 ? configured : 1;
    }

    private AiModel resolveQueueModel(Long modelId) {
        if (modelId != null) {
            try {
                AiModel model = aiModelService.getById(modelId);
                if (model != null && model.getStatus() != null && model.getStatus() == 1) {
                    return model;
                }
            } catch (Exception e) {
                log.warn("[ImageConsumer] 读取图片模型并发配置失败: modelId={}", modelId, e);
            }
        }

        AiModel defaultModel = aiModelService.getDefaultByType(MODEL_TYPE_IMAGE);
        if (defaultModel != null) {
            return defaultModel;
        }

        List<AiModel> imageModels = aiModelService.getListByType(MODEL_TYPE_IMAGE);
        return imageModels.isEmpty() ? null : imageModels.get(0);
    }

    private String resolveQueueName(Long modelId) {
        if (modelId == null) {
            throw new BusinessException("图片生成任务缺少 modelId，无法路由到模型队列");
        }
        return MODEL_QUEUE_PREFIX + modelId;
    }

    @PreDestroy
    public void shutdownWorkerExecutor() {
        workerExecutor.shutdownNow();
    }

    /**
     * 将远程图片 URL 下载到持久化存储（本地磁盘 / OSS），
     * 并替换 ImageItem 中的 URL 为永久可访问地址。
     */
    private void persistImageItems(ImageTask task) {
        List<ImageItem> items = imageGenerationService.listItems(task.getId());
        for (ImageItem item : items) {
            if (StrUtil.isNotBlank(item.getImageUrl())) {
                try {
                    String persistedUrl = mediaStorageService.downloadAndStore(item.getImageUrl(), "images");
                    item.setImageUrl(persistedUrl);
                    imageGenerationService.updateItem(item);
                    log.info("[ImageConsumer] 图片已持久化: {} -> {}", item.getId(), persistedUrl);
                } catch (Exception e) {
                    log.warn("[ImageConsumer] 图片持久化失败（保留原始 URL）: itemId={}, error={}",
                            item.getId(), e.getMessage());
                }
            }
        }
    }
}
