package com.stonewu.fusion.service.canvas.run.mcp;

import com.stonewu.fusion.mapper.canvas.CanvasMcpDecisionMapper;
import com.stonewu.fusion.entity.canvas.CanvasMcpDecisionEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.*;

@Service
@RequiredArgsConstructor
public class CanvasMcpReviewService {
    public record Decision(String id, String action, String note, long userId) {
    }

    public record View(String id, String name, String serverName, String toolName, Effect effect,
                       Idempotency idempotency,
                       String status, String error, long revision, ObjectNode arguments, ObjectNode received,
                       boolean remoteCapacityHeld, boolean cancelRequested, List<String> actions,
                       List<Decision> history) {
    }

    private final CanvasMcpDecisionMapper decisions;
    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasMcpRepository tasks;
    private final CanvasRunSources sources;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View detail(String runId, String nodeId, CanvasActor actor) {
        var run = runs.run(runId, false);
        access.requireView(documents.require(run.canvasId(), false), actor);
        var job = task(runId, nodeId, false);
        job.input().sources().all().forEach(value -> sources.authorize(value, actor, false));
        var contract = job.input().contract();
        var definition = contract.definition();
        var history = decisions.selectList(
                        Wrappers.<CanvasMcpDecisionEntity>lambdaQuery().eq(CanvasMcpDecisionEntity::getJobId, job.id())
                                .orderByAsc(CanvasMcpDecisionEntity::getCreatedAt, CanvasMcpDecisionEntity::getId)).stream()
                .map(row -> new Decision(row.getId(), row.getAction(), row.getNote(), row.getUserId())).toList();
        return new View(job.id(), contract.name(), definition.tool().serverName(), definition.tool().toolName(),
                definition.effect(), definition.idempotency(),
                job.status(), job.error(), job.revision(), job.input().arguments(), job.result(), job.permitHeld(),
                job.cancelled(),
                List.of(), history);
    }

    private Job task(String runId, String nodeId, boolean lock) {
        var node = runs.node(nodeId, false);
        if (!node.runId().equals(runId) || !"MCP".equals(node.externalKind()) || node.externalId() == null) {
            throw CanvasRunException.invalid("此节点没有对应的 MCP 调用");
        }
        return tasks.get(node.externalId(), lock);
    }
}
