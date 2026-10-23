package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;

import java.util.Map;

import static com.stonewu.fusion.service.canvas.agent.CanvasAgentSchema.*;

public final class CanvasImportAwaitTool extends CanvasAgentTool implements CanvasExternalTool {
    private final CanvasAgentImports imports;

    public CanvasImportAwaitTool(CanvasAgentImports imports, CanvasJson json) {
        super("await_canvas_import", "等待素材导入",
                "持久化等待本画布素材导入结束，不阻塞对话或重复下载。完成后使用 place_canvas_import 放入画布。",
                schema(Map.of("importId", string("导入任务 ID")), "importId"), true, json,
                (input, context) -> imports.await(text(input, "importId"), context));
        this.imports = imports;
    }

    @Override
    public Object poll(ObjectNode input, CanvasAgentScope scope, CanvasActor actor, AiModel model) {
        return imports.poll(text(input, "importId"), scope, actor);
    }
}
