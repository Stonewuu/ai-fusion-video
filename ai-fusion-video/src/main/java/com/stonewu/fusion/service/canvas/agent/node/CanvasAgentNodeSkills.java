package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;

import java.util.ArrayList;

public final class CanvasAgentNodeSkills {
    private CanvasAgentNodeSkills() {
    }

    public static AiChatReqVO configure(AiChatReqVO request, CanvasNode node) {
        String mode = node.config().path("skillMode").asText("INHERIT");
        boolean inherit = mode.equals("INHERIT");
        var names = new ArrayList<String>();
        node.config().path("skillNames").forEach(name -> names.add(name.asText()));
        if (inherit && !names.isEmpty()) {
            throw CanvasRunException.invalid("沿用智能体默认 Skill 时请清除本节点的指定列表，或切换为单独指定");
        }
        return request.setInheritAgentSkills(inherit).setEnabledSkills(names);
    }
}
