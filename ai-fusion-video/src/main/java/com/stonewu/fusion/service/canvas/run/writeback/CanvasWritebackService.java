package com.stonewu.fusion.service.canvas.run.writeback;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunInputs;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Service
@RequiredArgsConstructor
public class CanvasWritebackService {
    public enum Action {APPROVE, SKIP}

    public record Decision(@NotNull Long expectedRevision, @NotBlank String proposalHash, @NotNull Action action,
                           @NotBlank @Size(max = 1000) String note, Boolean acceptReset,
                           @NotNull @Valid GenerationClientScope scope) {
    }

    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasWritebackPreview previews;
    private final ResourceCommandService commands;
    private final ResourceQueryService resources;
    private final CanvasRunSources sources;
    private final CanvasRunInputs inputs;
    private final CanvasJson json;
    private final ResourceAgentFence agents;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#runId")
    public void decide(String runId, String nodeRunId, Decision request, CanvasActor actor) {
        decide(runId, nodeRunId, request, actor, null);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasRun", key = "#runId")
    public void decide(String runId, String nodeRunId, Decision request, CanvasActor actor,
                       ResourceCommandContext context) {
        request.scope().require(actor);
        var observed = runs.run(runId, false);
        runs.lockActor(actor);
        var document = documents.require(observed.canvasId(), true);
        access.requireEdit(document, actor);
        access.requireExecute(document, actor);
        if (context != null) {
            if (!actor.equals(context.actor()) || !Objects.equals(observed.canvasId(), context.canvasId())) {
                throw CanvasRunException.invalid("写回上下文与原画布不一致");
            }
            agents.requireActive(context, document.projectId());
        }
        if (!observed.actor().equals(actor)) {
            throw new CanvasRunException(403, "RUN_CONTROL_DENIED", "只能确认自己发起的业务写入");
        }
        var run = runs.run(runId, true);
        var node = runs.node(nodeRunId, true);
        if (!node.runId().equals(runId) || node.plan().node().kind() != CanvasNodeKind.WRITEBACK) {
            throw CanvasRunException.invalid("此节点不是当前运行的业务写入");
        }
        var proposal = node.checkpoint().path("writeback");
        if (!proposal.path("hash").asText().equals(request.proposalHash())) {
            throw CanvasRunException.conflict("确认内容与原提案不一致，请重新查看");
        }
        var previous = node.checkpoint().path("writebackDecision");
        String decisionHash = json.hash(request);
        if (node.status().terminal() && previous.path("requestHash").asText().equals(decisionHash)) {
            node.inputs().all().forEach(value -> sources.authorize(value, actor, false));
            return;
        }
        if (run.cancelRequested() || run.status().terminal() || !run.deadlineAt().isAfter(Instant.now())) {
            throw CanvasRunException.fenced();
        }
        if (node.status() != State.WAITING_CONFIRMATION || node.revision() != request.expectedRevision()) {
            throw CanvasRunException.conflict("写入等待状态已变化，请刷新后重试");
        }
        if (request.note() == null || request.note().isBlank() || request.note().length() > 1000) {
            throw CanvasRunException.invalid("请填写本次处理说明（最多 1000 字）");
        }
        var checkpoint = node.checkpoint().deepCopy();
        var decision = json.object(
                Map.of("action", request.action(), "note", request.note().strip(), "actorUserId", actor.userId(),
                        "expectedRevision", node.revision(), "requestHash", decisionHash, "at",
                        Instant.now().toString()));
        Ports output = Ports.empty();
        if (request.action() == Action.APPROVE) {
            if (proposal.path("dangerous").asBoolean() && !Boolean.TRUE.equals(request.acceptReset())) {
                throw CanvasRunException.invalid("请明确确认重置现有剧本解析内容及分集、场次");
            }
            previews.requireCurrent(run, node);
            var receipt = commands.execute(previews.command(node),
                    context == null ? ResourceCommandContext.user(actor, run.canvasId()) : context);
            var ref = node.inputs().at("target").getFirst().resource();
            var updated = resources.get(ref.type(), ref.id(), actor);
            // 此写入节点只修改分集字段，不修改镜头集合；保留原计划为后续合成冻结的镜头视频。
            var original = node.inputs().at("target").getFirst().data();
            if (original.has("_canvasVideos")) {
                updated.data().set("_canvasVideos", original.get("_canvasVideos"));
            }
            output = inputs.publish(node, new Ports(Map.of("resource", List.of(sources.business(updated)))));
            decision.put("commandId", receipt.commandId()).put("resourceVersion", updated.header().version());
        }
        checkpoint.set("writebackDecision", decision);
        runs.updateNode(node, request.action() == Action.APPROVE ? State.SUCCEEDED : State.CANCELLED, node.inputs(),
                output,
                checkpoint, "WRITEBACK", null,
                request.action() == Action.SKIP ? "已选择不写入业务内容，下游分支停止" : null, 0);
        runs.changeRun(runId, State.RUNNING, null, "CANVAS_RUN_WRITEBACK",
                Map.of("nodeRunId", nodeRunId, "decision", decision));
    }

    public static ObjectNode publicView(NodeRun node) {
        if (node.plan().node().kind() != CanvasNodeKind.WRITEBACK || !(node.checkpoint()
                .get("writeback") instanceof ObjectNode proposal)) {
            return null;
        }
        var view = proposal.deepCopy();
        if (node.checkpoint().get("writebackDecision") instanceof ObjectNode decision) {
            var publicDecision = decision.deepCopy();
            publicDecision.remove("requestHash");
            view.set("decision", publicDecision);
        }
        return view;
    }
}
