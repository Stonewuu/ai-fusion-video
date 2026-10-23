package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import io.agentscope.core.message.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.agent.CanvasAgentSchema.*;

public final class CanvasInspectTool extends CanvasAgentTool implements CanvasExternalTool {
    private final CanvasAgentMediaInspection inspection;
    private final CanvasJson json;

    public CanvasInspectTool(CanvasAgentMediaInspection inspection, CanvasJson json) {
        super("inspect_canvas_media", "观察画布媒体",
                "把已授权节点媒体实际投影为模型多模态输入。preview 为图片缩略图或视频的一张抽样帧，明确返回采样时间及观察限制；original 提交原文件。"
                        + "模型不支持该输入会明确报错。输入 nodeId 和端口；观察运行结果还需 canvasRunId、outputId。准备私有预览时持久化等待，完成后自动继续。",
                schema(Map.of("nodeId", string("节点 ID"), "port", string("媒体输出端口"), "canvasRunId",
                        string("可选的运行 ID"),
                        "outputId", string("运行中的具体输出 ID"), "mode",
                        choice("观察范围，默认 preview", "preview", "original")), "nodeId", "port"),
                true, json, inspection::inspect);
        this.inspection = inspection;
        this.json = json;
    }

    @Override
    public Object poll(ObjectNode input, CanvasAgentScope scope, CanvasActor actor, AiModel model) {
        return inspection.poll(input, scope, actor, model);
    }

    @Override
    public List<ContentBlock> projectOutput(String result) {
        var payload = json.read(result, ObjectNode.class);
        List<ContentBlock> blocks = new ArrayList<>();
        blocks.add(TextBlock.builder().text(result).build());
        if (!"success".equals(payload.path("status").asText())) {
            return List.copyOf(blocks);
        }
        for (JsonNode input : payload.path("modelInputs")) {
            var source = URLSource.builder().url(input.path("reference").asText())
                    .mimeType(input.path("mimeType").asText()).build();
            blocks.add(switch (input.path("type").asText()) {
                case "image" -> ImageBlock.builder().source(source).build();
                case "video" -> VideoBlock.builder().source(source).build();
                case "audio" -> AudioBlock.builder().source(source).build();
                case "file" -> DataBlock.builder().name(input.path("name").asText()).source(source).build();
                default -> throw new IllegalStateException("媒体观察结果类型无效");
            });
        }
        return List.copyOf(blocks);
    }
}
