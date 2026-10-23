package com.stonewu.fusion.service.canvas.run;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.canvas.CanvasExportEntity;
import com.stonewu.fusion.entity.canvas.CanvasMcpJobEntity;
import com.stonewu.fusion.entity.canvas.CanvasTextJobEntity;
import com.stonewu.fusion.mapper.canvas.CanvasAgentNodeJobMapper;
import com.stonewu.fusion.mapper.canvas.CanvasExportMapper;
import com.stonewu.fusion.mapper.canvas.CanvasMcpJobMapper;
import com.stonewu.fusion.mapper.canvas.CanvasTextJobMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;

/**
 * 授权失效后只读取原节点绑定任务的终态，不读取或发布结果。
 */
@Component
@RequiredArgsConstructor
public class CanvasRunOrphanSettlement {
    private final CanvasAgentNodeRepository agentNodes;
    private final CanvasAgentNodeJobMapper agents;
    private final CanvasExportMapper exports;
    private final CanvasMcpJobMapper mcp;
    private final CanvasTextJobMapper text;
    private final GenerationSubmissionMapper generations;
    private final MediaJobMapper media;

    public boolean ended(NodeRun node) {
        return switch (node.externalKind()) {
            case "AGENT" -> {
                agentNodes.requestCancel(node.id());
                yield agents.ended(node.externalId(), node.id()) != null;
            }
            case "EXPORT" -> {
                exports.update(
                        Wrappers.<CanvasExportEntity>lambdaUpdate().eq(CanvasExportEntity::getId, node.externalId())
                                .eq(CanvasExportEntity::getNodeRunId, node.id())
                                .in(CanvasExportEntity::getStatus, List.of("QUEUED", "RUNNING"))
                                .set(CanvasExportEntity::getCancelRequested, true));
                yield exports.exists(
                        Wrappers.<CanvasExportEntity>lambdaQuery().eq(CanvasExportEntity::getId, node.externalId())
                                .eq(CanvasExportEntity::getNodeRunId, node.id())
                                .in(CanvasExportEntity::getStatus, List.of("SUCCEEDED", "FAILED", "CANCELLED")));
            }
            case "MCP" -> {
                mcp.update(Wrappers.<CanvasMcpJobEntity>lambdaUpdate().eq(CanvasMcpJobEntity::getId, node.externalId())
                        .eq(CanvasMcpJobEntity::getNodeRunId, node.id())
                        .set(CanvasMcpJobEntity::getCancelRequested, true));
                yield mcp.exists(
                        Wrappers.<CanvasMcpJobEntity>lambdaQuery().eq(CanvasMcpJobEntity::getId, node.externalId())
                                .eq(CanvasMcpJobEntity::getNodeRunId, node.id())
                                .eq(CanvasMcpJobEntity::getPermitHeld, false)
                                .in(CanvasMcpJobEntity::getStatus, List.of("SUCCEEDED", "FAILED", "CANCELLED")));
            }
            case "TEXT" -> text.exists(
                    Wrappers.<CanvasTextJobEntity>lambdaQuery().eq(CanvasTextJobEntity::getId, node.externalId())
                            .eq(CanvasTextJobEntity::getNodeRunId, node.id())
                            .eq(CanvasTextJobEntity::getPermitHeld, false)
                            .in(CanvasTextJobEntity::getStatus, List.of("SUCCEEDED", "FAILED", "CANCELLED")));
            case "GENERATION" -> generations.endedNode(node.externalId(), node.id()) != null;
            case "MEDIA" -> media.endedCanvas(node.externalId(), node.id()) != null;
            default -> false;
        };
    }
}
