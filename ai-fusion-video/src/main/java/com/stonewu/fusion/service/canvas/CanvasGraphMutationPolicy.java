package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;

/**
 * 图结构校验通过后，在同一事务中校验外部业务、模型、媒体约束。人工和 Agent 共用此边界。
 */
public interface CanvasGraphMutationPolicy {
    void validate(CanvasDocument document, CanvasGraph before, CanvasGraphMutation mutation, CanvasActor actor);
}
