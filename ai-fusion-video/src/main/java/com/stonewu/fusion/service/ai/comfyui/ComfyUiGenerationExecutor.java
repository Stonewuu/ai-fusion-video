package com.stonewu.fusion.service.ai.comfyui;

import cn.hutool.core.util.StrUtil;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiDownloadedFile;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiJobResult;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiNativeClient;
import com.stonewu.fusion.service.storage.MediaStorageService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

/** Shared execution orchestration for ComfyUI image and video strategies. */
@Service
@RequiredArgsConstructor
public class ComfyUiGenerationExecutor {

    private final ApiConfigService apiConfigService;
    private final ComfyUiWorkflowService workflowService;
    private final ComfyUiWorkflowDocumentService documentService;
    private final ComfyUiWorkflowRenderer renderer;
    private final ComfyUiInputResourceService inputResourceService;
    private final ComfyUiOutputResolver outputResolver;
    private final ComfyUiNativeClient nativeClient;
    private final MediaStorageService mediaStorageService;

    public ComfyUiExecutionContext resolveActiveContext(AiModel model) {
        if (model == null || model.getComfyuiWorkflowId() == null) {
            throw new BusinessException(400, "ComfyUI 模型未绑定工作流");
        }
        ComfyUiWorkflow workflow = workflowService.requireWorkflow(model.getComfyuiWorkflowId());
        if (workflow.getActiveVersionId() == null) {
            throw new BusinessException(400, "ComfyUI 工作流没有已发布版本");
        }
        return resolveContext(model, workflow.getActiveVersionId());
    }

    public ComfyUiExecutionContext resolveContext(AiModel model, Long versionId) {
        if (model == null || model.getComfyuiWorkflowId() == null) {
            throw new BusinessException(400, "ComfyUI 模型未绑定工作流");
        }
        if (versionId == null) {
            throw new BusinessException(400, "ComfyUI 任务未固化工作流版本");
        }
        ComfyUiWorkflow workflow = workflowService.requireWorkflow(model.getComfyuiWorkflowId());
        ComfyUiWorkflowVersion version = workflowService.requireVersion(versionId);
        if (!workflow.getId().equals(version.getWorkflowId())) {
            throw new BusinessException(400, "任务工作流版本与模型不匹配");
        }
        if (!Boolean.TRUE.equals(version.getPublished())) {
            throw new BusinessException(400, "任务不能执行未发布的 ComfyUI 工作流版本");
        }
        ApiConfig apiConfig = apiConfigService.getById(model.getApiConfigId());
        if (apiConfig == null || !ComfyUiWorkflowService.PLATFORM.equalsIgnoreCase(apiConfig.getPlatform())) {
            throw new BusinessException(400, "ComfyUI API 配置不存在或平台不匹配");
        }
        if (!apiConfig.getId().equals(workflow.getApiConfigId())) {
            throw new BusinessException(400, "ComfyUI 工作流与模型 API 配置不一致");
        }
        return new ComfyUiExecutionContext(model, apiConfig, workflow, version);
    }

    public ComfyUiExecutionContext resolveTestContext(Long versionId) {
        ComfyUiWorkflowVersion version = workflowService.requireVersion(versionId);
        if (!Integer.valueOf(ComfyUiWorkflowVersion.VALIDATION_VALID)
                .equals(version.getValidationStatus())) {
            throw new BusinessException(400, "请先完成目标 ComfyUI 在线依赖校验");
        }
        ComfyUiWorkflow workflow = workflowService.requireWorkflow(version.getWorkflowId());
        ApiConfig apiConfig = apiConfigService.getById(workflow.getApiConfigId());
        if (apiConfig == null || !Integer.valueOf(1).equals(apiConfig.getStatus())
                || !ComfyUiWorkflowService.PLATFORM.equalsIgnoreCase(apiConfig.getPlatform())) {
            throw new BusinessException(400, "ComfyUI API 配置不存在、已禁用或平台不匹配");
        }
        AiModel syntheticModel = AiModel.builder()
                .name(workflow.getName())
                .code(workflow.getCode())
                .modelType(workflow.getModelType())
                .apiConfigId(workflow.getApiConfigId())
                .comfyuiWorkflowId(workflow.getId())
                .modelProtocol(ComfyUiWorkflowService.PLATFORM)
                .status(1)
                .build();
        return new ComfyUiExecutionContext(syntheticModel, apiConfig, workflow, version);
    }

    public ComfyUiPreparedSubmission prepare(ComfyUiExecutionContext context,
                                              String taskKey,
                                              Map<String, Object> rawValues) {
        Map<String, Object> values = new LinkedHashMap<>(rawValues == null ? Map.of() : rawValues);
        uploadBoundImages(context, taskKey, values);
        ObjectNode prompt = renderer.render(context.model().getModelType(), context.version(), values);
        return new ComfyUiPreparedSubmission(context, UUID.randomUUID().toString(), prompt);
    }

    public void submit(ComfyUiPreparedSubmission submission) {
        nativeClient.submitPrompt(
                submission.context().apiConfig(), submission.prompt(), submission.promptId());
    }

    public ComfyUiJobResult waitForJob(ComfyUiExecutionContext context,
                                       String promptId,
                                       long pollIntervalMillis,
                                       long timeoutMillis) {
        long interval = Math.max(500L, pollIntervalMillis);
        long deadline = System.currentTimeMillis() + Math.max(10_000L, timeoutMillis);
        while (System.currentTimeMillis() <= deadline) {
            ComfyUiJobResult job = nativeClient.getJob(context.apiConfig(), promptId);
            if (job.completed()) return job;
            if (job.failed()) {
                String error = job.executionError() == null
                        ? "未知错误" : job.executionError().toString();
                throw new BusinessException(502, "ComfyUI 工作流执行失败: " + error);
            }
            sleep(interval);
        }
        throw new BusinessException(504, "ComfyUI 工作流轮询超时: " + promptId);
    }

    public List<ComfyUiStoredOutput> storeOutputs(ComfyUiExecutionContext context,
                                                   ComfyUiJobResult job) {
        List<ComfyUiOutputBinding> bindings = documentService.parseOutputBindings(
                context.model().getModelType(), context.version().getApiWorkflowJson(),
                context.version().getOutputBindingsJson());
        List<ComfyUiRemoteOutput> remoteOutputs = outputResolver.resolve(job, bindings);
        List<ComfyUiStoredOutput> stored = new ArrayList<>();
        for (ComfyUiRemoteOutput output : remoteOutputs) {
            try (ComfyUiDownloadedFile file = nativeClient.downloadOutput(
                    context.apiConfig(), output.filename(), output.subfolder(), output.type())) {
                String url = mediaStorageService.storeFile(
                        file.path(), storageDirectory(output), file.extension());
                stored.add(new ComfyUiStoredOutput(
                        output.mediaType(), output.role(), url, file.size()));
            } catch (IOException e) {
                throw new BusinessException(500,
                        "清理 ComfyUI 临时输出失败: " + StrUtil.blankToDefault(e.getMessage(), "I/O error"));
            }
        }
        return List.copyOf(stored);
    }

    public boolean cancel(ComfyUiExecutionContext context, String promptId) {
        return nativeClient.cancelJob(context.apiConfig(), promptId);
    }

    private void uploadBoundImages(ComfyUiExecutionContext context,
                                   String taskKey,
                                   Map<String, Object> values) {
        List<ComfyUiInputBinding> bindings = documentService.parseInputBindings(
                context.model().getModelType(), context.version().getApiWorkflowJson(),
                context.version().getInputBindingsJson());
        Set<String> imageFields = new LinkedHashSet<>();
        for (ComfyUiInputBinding binding : bindings) {
            if ("uploaded_video".equals(binding.valueType())
                    || "uploaded_audio".equals(binding.valueType())) {
                throw new BusinessException(400,
                        "Native API 第一版不支持视频/音频文件上传绑定: " + binding.businessField());
            }
            if ("uploaded_image".equals(binding.valueType())) {
                imageFields.add(binding.businessField());
            }
        }
        for (String field : imageFields) {
            Object raw = values.get(field);
            if (raw == null) continue;
            List<String> sources = toStringList(raw, field);
            values.put(field, inputResourceService.uploadImages(
                    context.apiConfig(), taskKey, field, sources));
        }
    }

    private List<String> toStringList(Object raw, String field) {
        if (raw instanceof List<?> list) {
            return list.stream().map(value -> {
                if (value == null || value.toString().isBlank()) {
                    throw new BusinessException(400, "ComfyUI 图片输入包含空值: " + field);
                }
                return value.toString();
            }).toList();
        }
        if (raw.toString().isBlank()) {
            throw new BusinessException(400, "ComfyUI 图片输入不能为空: " + field);
        }
        return List.of(raw.toString());
    }

    private String storageDirectory(ComfyUiRemoteOutput output) {
        return switch (output.mediaType()) {
            case "image" -> "images";
            case "video" -> "videos";
            case "audio" -> "audio";
            default -> "files/comfyui";
        };
    }

    private void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new BusinessException(500, "ComfyUI 工作流轮询被中断");
        }
    }
}
