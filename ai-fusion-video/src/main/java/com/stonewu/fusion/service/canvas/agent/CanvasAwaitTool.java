package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;

import java.util.Map;

import static com.stonewu.fusion.service.canvas.agent.CanvasAgentSchema.*;

public final class CanvasAwaitTool extends CanvasAgentTool implements CanvasExternalTool {
    private final CanvasAgentAwaitService waits;

    public CanvasAwaitTool(CanvasAgentAwaitService waits, CanvasJson json) {
        super("await_canvas_run", "等待画布任务",
                "持久化等待当前画布运行完成或出现候选、写回确认、核实等待点。等待期间释放工具线程，关闭页面不会停止已受理任务；请不要用查询循环代替等待。",
                schema(Map.of("canvasRunId", string("画布运行 ID")), "canvasRunId"), true, json,
                (input, context) -> waits.await(text(input, "canvasRunId"), context));
        this.waits = waits;
    }

    @Override
    public Object poll(ObjectNode input, CanvasAgentScope scope, CanvasActor actor, AiModel model) {
        return waits.poll(text(input, "canvasRunId"), scope, actor);
    }
}
