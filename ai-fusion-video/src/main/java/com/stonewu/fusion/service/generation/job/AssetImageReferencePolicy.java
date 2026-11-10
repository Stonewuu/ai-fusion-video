package com.stonewu.fusion.service.generation.job;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;

/**
 * 一个资产执行任务只处理一个子资产，后续调用必须沿用首次受理的参考图及媒体版本。
 */
@Component
@RequiredArgsConstructor
public class AssetImageReferencePolicy {
    private final AgentRunMapper runs;
    private final GenerationSubmissionMapper submissions;
    private final GenerationRequestRepository requests;
    private final ReferenceImageTransportService references;
    private final CanvasJson json;

    /** 调用方已通过执行身份校验并持有运行行锁，校验与新任务受理在同一事务完成。 */
    @Transactional(propagation = Propagation.MANDATORY)
    public void enforce(ImageTask task, ResourceCommandContext context) {
        if (!"agent".equals(context.origin())) {
            return;
        }
        var run = runs.selectByRunId(context.agentRunId());
        if (!"asset_image_executor".equals(run.getAgentType())) {
            return;
        }
        var first = submissions.selectAssetImageReferenceSnapshot(requests.scope(context), context.agentRunId());
        if (first == null) {
            return;
        }
        if (first.inputJson() == null) {
            throw missingSnapshot();
        }
        var original = json.read(first.inputJson(), ReferenceInput.class);
        if (original.inputMediaVersions() == null) {
            throw missingSnapshot();
        }
        var expected = references.parseJsonInputs(original.refImageUrls(), "refImageUrls");
        var actual = references.parseJsonInputs(task.getRefImageUrls(), "refImageUrls");
        if (!actual.equals(expected)) {
            throw new GenerationJobException(409, "ASSET_IMAGE_REFERENCES_CHANGED",
                    "同一子资产任务的参考图不能增加、移除、替换或调整顺序，请沿用首次请求的 imageUrls；无法继续时保留失败原因并停止");
        }
        if (task.getInputMediaVersions() != null
                && !task.getInputMediaVersions().equals(original.inputMediaVersions())) {
            throw new GenerationJobException(409, "GENERATION_INPUT_CHANGED", "不能更改同一子资产任务已固定的参考图版本");
        }
        // 后续的媒体校验使用首次版本，不能在失败重试或跨实例恢复时读取新版本代替。
        task.setInputMediaVersions(Map.copyOf(original.inputMediaVersions()));
    }

    private GenerationJobException missingSnapshot() {
        return new GenerationJobException(409, "GENERATION_INPUT_SNAPSHOT_MISSING",
                "此资产任务缺少首次生成的输入快照，已停止提交，请重新检查任务");
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    private record ReferenceInput(String refImageUrls, Map<String, Integer> inputMediaVersions) {
    }
}
