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
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.*;

/**
 * Shared execution orchestration for ComfyUI image and video strategies.
 */
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
        return resolveContext(model, versionId,
                model == null ? null : apiConfigService.getById(model.getApiConfigId()));
    }

    public ComfyUiExecutionContext resolveContext(AiModel model, Long versionId, ApiConfig apiConfig) {
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
        if (apiConfig == null || !ComfyUiWorkflowService.PLATFORM.equalsIgnoreCase(apiConfig.getPlatform())) {
            throw new BusinessException(400, "ComfyUI API 配置不存在或平台不匹配");
        }
        if (!apiConfig.getId().equals(workflow.getApiConfigId())) {
            throw new BusinessException(400, "ComfyUI 工作流与模型 API 配置不一致");
        }
        return new ComfyUiExecutionContext(model, apiConfig, workflow, version);
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

    public ComfyUiJobResult query(ComfyUiExecutionContext context, String promptId) {
        return nativeClient.getJob(context.apiConfig(), promptId);
    }

    public List<ComfyUiStoredOutput> storeOutputs(ComfyUiExecutionContext context,
                                                  ComfyUiJobResult job, OutputWriter writer) {
        List<ComfyUiOutputBinding> bindings = documentService.parseOutputBindings(
                context.model().getModelType(), context.version().getApiWorkflowJson(),
                context.version().getOutputBindingsJson());
        List<ComfyUiRemoteOutput> remoteOutputs = outputResolver.resolve(job, bindings);
        List<ComfyUiStoredOutput> stored = new ArrayList<>();
        for (int index = 0; index < remoteOutputs.size(); index++) {
            ComfyUiRemoteOutput output = remoteOutputs.get(index);
            try (ComfyUiDownloadedFile file = nativeClient.downloadOutput(
                    context.apiConfig(), output.filename(), output.subfolder(), output.type())) {
                String url = writer.store(output, file, index);
                stored.add(new ComfyUiStoredOutput(
                        output.mediaType(), output.role(), url, file.size()));
            } catch (IOException e) {
                throw new BusinessException(500,
                        "接收或清理 ComfyUI 输出失败: " + StrUtil.blankToDefault(e.getMessage(), "I/O error"));
            }
        }
        return List.copyOf(stored);
    }

    @FunctionalInterface
    public interface OutputWriter {
        String store(ComfyUiRemoteOutput output, ComfyUiDownloadedFile file, int index) throws IOException;
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
        validateMediaCoverage(bindings, values);
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
            if (raw == null) {
                continue;
            }
            List<String> sources = toStringList(raw, field);
            values.put(field, inputResourceService.uploadImages(
                    context.apiConfig(), taskKey, field, sources));
        }
    }

    private void validateMediaCoverage(List<ComfyUiInputBinding> bindings, Map<String, Object> values) {
        for (String field : List.of("referenceImages", "firstFrame", "lastFrame",
                "referenceVideos", "referenceAudios")) {
            Object raw = values.get(field);
            if (raw == null) {
                continue;
            }
            int count = raw instanceof List<?> items ? items.size() : 1;
            Set<Integer> covered = new HashSet<>();
            boolean wholeList = false;
            for (ComfyUiInputBinding binding : bindings) {
                if (!field.equals(binding.businessField())) {
                    continue;
                }
                if (binding.index() == null) {
                    if ("string_list".equals(binding.valueType())) {
                        wholeList = true;
                    } else {
                        covered.add(0);
                    }
                } else {
                    if (binding.index() >= count) {
                        throw new BusinessException(400, "工作流输入 " + field + " 缺少索引 " + binding.index());
                    }
                    covered.add(binding.index());
                }
            }
            if (wholeList) {
                continue;
            }
            for (int index = 0; index < count; index++) {
                if (!covered.contains(index)) {
                    throw new BusinessException(400, "ComfyUI 工作流输入 " + field + " 的索引 " + index
                            + "（第 " + (index + 1) + " 项）未绑定，不能忽略素材；请扩展工作流绑定或减少输入");
                }
            }
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

}
