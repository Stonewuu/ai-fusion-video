package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelMultimodalCapabilities;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.media.preview.MediaPreviewData.Request;
import com.stonewu.fusion.service.media.preview.MediaPreviewService;
import io.agentscope.core.tool.ToolSuspendException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.agent.CanvasAgentSchema.text;

@Service
@RequiredArgsConstructor
public class CanvasAgentMediaInspection {
    private final CanvasAgentAccess access;
    private final CanvasQueryService canvases;
    private final CanvasRunService runs;
    private final CanvasRunSources sources;
    private final MediaResourceQueries media;
    private final MediaPreviewService previews;

    public Object inspect(ObjectNode input, ToolExecutionContext context) {
        var session = access.require(context);
        var result = poll(input, session.scope(), session.context().actor(), context.getModel());
        if (result != null) {
            return result;
        }
        throw new ToolSuspendException("正在准备私有预览，完成后自动交给模型观察。");
    }

    public Object poll(ObjectNode input, CanvasAgentScope scope, CanvasActor actor, AiModel model) {
        String nodeId = text(input, "nodeId");
        String port = text(input, "port");
        String mode = input.hasNonNull("mode") ? text(input, "mode") : "preview";
        if (!List.of("preview", "original").contains(mode)) {
            throw new BusinessException(400, "观察模式必须为 preview 或 original");
        }
        if (model == null) {
            throw new BusinessException(409, "本轮模型配置不可用，无法观察媒体");
        }
        var snapshot = canvases.snapshot(scope.canvasId(), actor);
        Value value;
        if (input.hasNonNull("canvasRunId")) {
            var run = runs.view(text(input, "canvasRunId"), actor);
            if (!run.canvasId().equals(scope.canvasId())) {
                throw new BusinessException(403, "不能观察其他画布的运行结果");
            }
            String outputId = text(input, "outputId");
            if (outputId == null) {
                throw new BusinessException(400, "观察运行结果必须指定真实 outputId");
            }
            value = run.nodes().stream().filter(node -> node.nodeId().equals(nodeId))
                    .flatMap(node -> node.outputs().all().stream()).filter(output -> output.id().equals(outputId))
                    .findFirst()
                    .orElseThrow(() -> new BusinessException(404, "节点运行输出不存在或已失去访问权限"));
        } else {
            var node = snapshot.graph().nodes().stream().filter(item -> item.id().equals(nodeId)).findFirst()
                    .orElseThrow(() -> new BusinessException(404, "节点不在当前画布中"));
            var outputs = sources.freeze(snapshot.document(), node, actor, Set.of(port)).at(port);
            if (outputs.size() != 1) {
                throw new BusinessException(400, "请指定一个确定的媒体端口或运行 outputId");
            }
            value = outputs.getFirst();
        }
        if (value.mediaId() == null) {
            throw new BusinessException(422, "所选内容不是媒体，请使用节点正文读取工具");
        }
        sources.authorize(value, actor, true);
        var source = media.get(value.mediaId(), actor);
        String observedId = source.id();
        long sample = 0;
        String coverage = "original_input";
        String limitation = "文件已作为模型输入；描述应以模型实际可读内容为准。";
        if (mode.equals("preview")) {
            if (!Set.of(MediaKind.IMAGE, MediaKind.VIDEO).contains(source.content().kind())) {
                throw new BusinessException(422, "音频和文档请明确选择 original，并使用支持此输入类型的模型");
            }
            AiModelMultimodalCapabilities.managedTransport(model, "image/jpeg");
            var preview = previews.ensure(source.id(), new Request(null), actor);
            if (List.of("FAILED", "CANCELLED").contains(preview.status())) {
                throw new BusinessException(422, "预览准备失败，请在画布中查看并重试：" + preview.error());
            }
            if (!"SUCCEEDED".equals(preview.status())) {
                return null;
            }
            observedId = preview.previewId();
            sample = preview.sampleAtMs();
            coverage = source.content().kind() == MediaKind.VIDEO ? "video_sample_frame" : "image_thumbnail";
            limitation = source.content().kind() == MediaKind.VIDEO
                    ? "只观察了指定毫秒的单帧，未检查完整视频、运动连续性或声音。" : "观察的是缩略图；需要细节时应显式读取原图。";
        } else if (source.content().byteSize() > 64L * 1024 * 1024) {
            throw new BusinessException(422, "原文件超过单次观察的 64 MiB 上限，请先提取需要的片段或预览");
        }
        var observed = media.get(observedId, actor);
        String transport = AiModelMultimodalCapabilities.managedTransport(model, observed.content().mimeType());
        String type = AiModelMultimodalCapabilities.validateUpload(model, observed.content().mimeType(), transport);
        if (source.content().kind() == MediaKind.VIDEO && mode.equals("original")) {
            limitation = "视频原文件已提交；实际采样由模型供应商决定，不能声称逐帧检查了全片。";
        }
        return Map.of("status", "success", "source", source, "observation",
                Map.of("coverage", coverage, "sampleAtMs", sample,
                        "limitation", limitation, "observedMediaId", observed.id(), "observedMediaVersion",
                        observed.version()),
                "modelInputs", List.of(Map.of("type", type, "reference", MediaReference.of(observed.id()), "mimeType",
                        observed.content().mimeType(), "name", observed.name())));
    }
}
