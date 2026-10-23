package com.stonewu.fusion.service.generation.image.strategy.comfyui;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiBoundInputs;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiGenerationExecutor;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.image.strategy.ImageGenerationStrategy;
import com.stonewu.fusion.service.generation.job.*;
import com.stonewu.fusion.service.media.GenerationMediaService;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Image generation through an administrator-published ComfyUI workflow.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ComfyUiImageStrategy implements ImageGenerationStrategy {

    private static final long DEFAULT_POLL_INTERVAL_MILLIS = 2_000L;
    private static final long DEFAULT_TIMEOUT_MILLIS = 25L * 60L * 1_000L;

    private final AiModelService aiModelService;
    private final ImageGenerationService imageGenerationService;
    private final ComfyUiGenerationExecutor executor;
    private final GenerationMediaService media;

    @Override
    public String getName() {
        return ComfyUiWorkflowService.PLATFORM;
    }


    @Override
    public int batchSize(ImageTask task, AiModel model, ApiConfig apiConfig) {
        return task.getCount();
    }

    @Override
    public PreparedGenerationCall prepare(ImageTask task, AiModel model, ApiConfig apiConfig) {
        var context = executor.resolveContext(model, task.getWorkflowVersionId(), apiConfig);
        int[] size = resolveDefaultSize(model, task);
        var values = imageValues(task.getPrompt(), size[0], size[1], task.getCount(),
                parseRefImageUrls(task.getRefImageUrls()), randomSeed());
        ComfyUiBoundInputs.overlay(values, task.getWorkflowInputs(), task.getCount());
        var prepared = executor.prepare(context, GenerationLeaseScope.require().id(), values);
        return new PreparedGenerationCall(new GenerationPreparedRequest(prepared.promptId(), prepared.prompt()), () -> {
            executor.submit(prepared);
            return GenerationProviderReceipt.accepted(prepared.promptId());
        });
    }

    @Override
    public RemoteGenerationResult query(String platformTaskId, ImageTask task, AiModel model, ApiConfig apiConfig) {
        var context = executor.resolveContext(model, task.getWorkflowVersionId(), apiConfig);
        var job = executor.query(context, platformTaskId);
        if (!job.completed()) {
            return ProviderJobStates.result(job.status(), "ComfyUI 工作流未成功结束", 3, List::of);
        }
        String prefix = "comfy-" + GenerationLeaseScope.require().id() + "-";
        var stored = executor.storeOutputs(context, job, (output, file, index) ->
                media.file(GenerationKind.IMAGE, task.getId(), prefix + index, file.path(),
                        switch (output.mediaType()) {
                            case "image" -> MediaKind.IMAGE;
                            case "video" -> MediaKind.VIDEO;
                            default -> throw new BusinessException("ComfyUI 输出媒体类型未登记");
                        }));
        return RemoteGenerationResult.success(stored.stream()
                .filter(output -> "image".equals(output.mediaType()) && "primary".equals(output.role()))
                .map(output -> RemoteGenerationResult.Output.image(output.url())).toList());
    }

    @Override
    public boolean cancel(String platformTaskId, ImageTask task, AiModel model, ApiConfig apiConfig) {
        return executor.cancel(executor.resolveContext(model, task.getWorkflowVersionId(), apiConfig), platformTaskId);
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
        if (model == null || StrUtil.isBlank(model.getConfig())) {
            return fallback;
        }
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
