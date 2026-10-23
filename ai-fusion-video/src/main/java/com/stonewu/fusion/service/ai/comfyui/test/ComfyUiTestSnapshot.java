package com.stonewu.fusion.service.ai.comfyui.test;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiExecutionContext;

import java.util.Map;

/**
 * 草稿试运行不要求已发布模型；保存工作流定义，不虚构模型数据库标识。
 */
public record ComfyUiTestSnapshot(int schemaVersion, ComfyUiWorkflow workflow,
                                  ComfyUiWorkflowVersion version, Map<String, Object> inputs,
                                  Map<String, Integer> mediaVersions) {
    public ComfyUiExecutionContext execution(ApiConfig api) {
        if (schemaVersion != 1) {
            throw new IllegalStateException("不支持的 ComfyUI 试运行快照版本");
        }
        var model = AiModel.builder().name(workflow.getName()).code(workflow.getCode())
                .modelType(workflow.getModelType()).apiConfigId(workflow.getApiConfigId())
                .comfyuiWorkflowId(workflow.getId()).modelProtocol("comfyui").status(1).build();
        return new ComfyUiExecutionContext(model, api, workflow, version);
    }
}
