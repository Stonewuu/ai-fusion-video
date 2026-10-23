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
    public enum Action {USE_RECEIVED, CONFIRM_FINISHED}

    public record Review(@NotBlank @Pattern(regexp = "[A-Za-z0-9._:-]{1,128}") String requestId,
                         @NotNull Long expectedRevision, @NotNull Action action,
                         @NotBlank @Size(max = 2000) String note,
                         boolean providerFinished, @NotNull @Valid GenerationClientScope scope) {
    }

    public record Decision(String id, String action, String note, long userId) {
    }

    public record View(String id, String name, String serverName, String toolName, Effect effect,
                       Idempotency idempotency,
                       String status, String error, long revision, ObjectNode arguments, ObjectNode received,
                       boolean remoteCapacityHeld, boolean cancelRequested, List<Action> actions,
                       List<Decision> history) {
    }

    private final CanvasMcpDecisionMapper decisions;
    private final CanvasJson json;
    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasMcpRepository tasks;
    private final CanvasMcpContracts contracts;
    private final CanvasRunSources sources;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View detail(String runId, String nodeId, CanvasActor actor) {
        var run = runs.run(runId, false);
        var permissions = access.requireView(documents.require(run.canvasId(), false), actor);
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
                run.actor().equals(actor) && permissions.canExecute() ? actions(job, run) : List.of(), history);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = {"canvasRun", "canvasMcpJob"}, allEntries = true)
    public View review(String runId, String nodeId, Review request, CanvasActor actor) {
        request.scope().require(actor);
        runs.lockActor(actor);
        var observed = runs.run(runId, false);
        if (!access.requireView(documents.require(observed.canvasId(), true), actor).canExecute() || !observed.actor()
                .equals(actor)) {
            throw new CanvasRunException(403, "MCP_REVIEW_DENIED", "只能核实原账号与团队发起的 MCP 调用");
        }
        var run = runs.run(runId, true);
        var job = task(runId, nodeId, true);
        String hash = json.hash(Map.of("jobId", job.id(), "request", request));
        var previous = decisions.lockRequest(actor.userId(), request.requestId());
        if (previous != null) {
            if (!hash.equals(previous.getRequestHash())) {
                throw CanvasRunException.conflict("同一核实请求不能用于不同决定");
            }
            return json.read(previous.getReceiptJson(), View.class);
        }
        if (request.expectedRevision() == null || request.expectedRevision() != job.revision() || !actions(job,
                run).contains(request.action())) {
            throw CanvasRunException.conflict("MCP 原调用有新状态或回执，请刷新后重新核实");
        }
        if (request.note() == null || request.note().isBlank() || request.note().length() > 2000) {
            throw CanvasRunException.invalid("请记录此次核实依据");
        }
        if (request.action() == Action.USE_RECEIVED) {
            access.requireExecute(documents.require(run.canvasId(), false), actor);
            contracts.verify(job.input().contract(), actor.userId());
            job.input().sources().all().forEach(value -> sources.authorize(value, actor, true));
            tasks.finish(job, "SUCCEEDED", null, false);
        } else {
            if (!request.providerFinished()) {
                throw CanvasRunException.invalid("请明确核实原远端操作已结束；确认不会再次调用工具");
            }
            tasks.finish(job, job.cancelled() || run.cancelRequested() ? "CANCELLED" : "FAILED",
                    "已人工核实原 MCP 调用结束，未重新发送请求", false);
        }
        String id = UUID.randomUUID().toString();
        decisions.create(id, job.id(), actor.userId(), request.requestId(), hash, request.action().name(),
                request.note().trim());
        var receipt = detail(runId, nodeId, actor);
        decisions.update(Wrappers.<CanvasMcpDecisionEntity>lambdaUpdate().eq(CanvasMcpDecisionEntity::getId, id)
                .set(CanvasMcpDecisionEntity::getReceiptJson, json.write(receipt)));
        return receipt;
    }

    private List<Action> actions(Job job, Run run) {
        if (!List.of("NEEDS_REVIEW", "UNKNOWN_SUBMISSION").contains(job.status()) || run.status().terminal()) {
            return List.of();
        }
        List<Action> result = new ArrayList<>();
        if (job.result() != null && job.result().path("valid").asBoolean() && !job.cancelled() && !run.cancelRequested()
                && job.deadlineAt().isAfter(Instant.now()) && run.deadlineAt().isAfter(Instant.now())) {
            result.add(Action.USE_RECEIVED);
        }
        result.add(Action.CONFIRM_FINISHED);
        return List.copyOf(result);
    }

    private Job task(String runId, String nodeId, boolean lock) {
        var node = runs.node(nodeId, false);
        if (!node.runId().equals(runId) || !"MCP".equals(node.externalKind()) || node.externalId() == null) {
            throw CanvasRunException.invalid("此节点没有对应的 MCP 调用");
        }
        return tasks.get(node.externalId(), lock);
    }
}
