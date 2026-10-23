package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.mapper.canvas.CanvasAgentNodeJobMapper;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.entity.generation.GenerationSubmissionEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.ai.AiAgentRegistry;
import com.stonewu.fusion.controller.ai.vo.ToolConfirmationReqVO;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.run.AgentConfirmationService;
import com.stonewu.fusion.service.ai.run.AgentEventEnvelopeSanitizer;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScope;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import com.stonewu.fusion.service.generation.job.GenerationJobQueries;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

@Service
@RequiredArgsConstructor
public class CanvasAgentNodeQueries {
    public record Confirmation(String replyId, Instant expiresAt, JsonNode tools) {
    }

    public record Progress(String id, String parentId, String name, String status, String error, String summary,
                           boolean summaryTruncated, Instant startedAt, Confirmation confirmation) {
    }

    public record View(String nodeRunId, String conversationId, boolean canControl, int maxAgentRuns,
                       long unresolvedModelCalls, List<String> unresolvedModelRequestIds, List<Progress> tasks,
                       List<GenerationJobQueries.Summary> generations) {
    }

    private final CanvasRunService runs;
    private final CanvasAgentNodeRepository jobs;
    private final AgentRunRepository agents;
    private final AgentEventMapper events;
    private final AgentConfirmationService confirmations;
    private final AgentEventEnvelopeSanitizer sanitizer;
    private final AiAgentRegistry definitions;
    private final CanvasAgentNodeJobMapper nodes;
    private final AgentMessageMapper messages;
    private final GenerationSubmissionMapper submissions;
    private final CanvasJson json;
    private final GenerationJobQueries generations;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View detail(String runId, String nodeRunId, CanvasActor actor) {
        var view = runs.view(runId, actor);
        var node = view.nodes().stream()
                .filter(item -> item.id().equals(nodeRunId) && "AGENT".equals(item.externalKind())).findFirst()
                .orElseThrow(() -> new BusinessException(404, "此节点没有专业智能体任务"));
        if (node.contentUnavailable()) {
            throw new BusinessException(403, "专业任务的输入或结果已不可访问");
        }
        var job = jobs.find(nodeRunId, false);
        if (job == null || !job.canvasRunId().equals(runId) || !job.agentRunId().equals(node.externalId())) {
            throw new BusinessException(409, "专业节点的任务关联不一致");
        }
        var ids = nodes.agentRunIds(nodeRunId);
        if (ids.size() > 64) {
            throw new BusinessException(409, "专业任务数量超过协议上限");
        }
        List<Progress> progress = new ArrayList<>();
        for (String id : ids) {
            var task = agents.findRun(id);
            var definition = definitions.getByType(task.getAgentType());
            var texts = messages.selectPage(new Page<AgentMessage>(1, 1, false), Wrappers.<AgentMessage>lambdaQuery()
                            .select(AgentMessage::getId, AgentMessage::getContent).eq(AgentMessage::getRunId, id)
                            .eq(AgentMessage::getRole, "assistant").isNotNull(AgentMessage::getContent)
                            .orderByDesc(AgentMessage::getMessageOrder)).getRecords().stream().map(AgentMessage::getContent)
                    .toList();
            String text = texts.isEmpty() ? null : texts.getFirst();
            boolean truncated = text != null && text.length() > 8000;
            if (truncated) {
                text = text.substring(0, 8000);
            }
            Confirmation confirmation = null;
            if (view.canControl() && "WAITING_CONFIRMATION".equals(task.getStatus())) {
                var candidate = events.selectConfirmationCandidate(id, task.getWaitingReplyId());
                if (candidate == null) {
                    throw new BusinessException(409, "审批记录缺失，请刷新任务状态");
                }
                var payload = json.object(json.read(candidate.getPayloadJson(), JsonNode.class));
                var previews = json.read(payload.path("pendingToolCallsJson").asText(), ArrayNode.class);
                var suspended = json.read(payload.path("suspendedToolCallsJson").asText(), ArrayNode.class);
                for (var preview : previews) {
                    for (var call : suspended) {
                        if (call.path("id").asText().equals(preview.path("toolCallId").asText())) {
                            ((ObjectNode) preview).put("arguments", json.write(sanitizer.sanitize(call.path("input"))));
                        }
                    }
                }
                confirmation = new Confirmation(task.getWaitingReplyId(),
                        task.getWaitExpiresAt().toInstant(ZoneOffset.UTC), previews);
            }
            progress.add(new Progress(id, task.getParentRunId(),
                    definition == null ? task.getAgentType() : definition.getName(),
                    task.getStatus(), task.getErrorMessage(), text, truncated,
                    task.getStartedAt().toInstant(ZoneOffset.UTC), confirmation));
        }
        var uncertain = nodes.unresolvedModelIds(nodeRunId);
        var generationProgress = submissions.selectList(Wrappers.<GenerationSubmissionEntity>lambdaQuery()
                        .select(GenerationSubmissionEntity::getId).eq(GenerationSubmissionEntity::getCanvasNodeRunId, nodeRunId)
                        .orderByAsc(GenerationSubmissionEntity::getCreatedAt, GenerationSubmissionEntity::getId))
                .stream().map(GenerationSubmissionEntity::getId)
                .map(id -> generations.summary(id, actor)).toList();
        return new View(nodeRunId, job.conversationId(), view.canControl(), job.scope().maxAgentRuns(),
                uncertain.size(), List.copyOf(uncertain), List.copyOf(progress), generationProgress);
    }

    public Mono<Void> confirm(String runId, String nodeRunId, ToolConfirmationReqVO request, CanvasActor actor) {
        var view = runs.view(runId, actor);
        if (!view.canControl() || view.cancelRequested()) {
            throw new BusinessException(403, "只能由此运行的发起人在原账号与团队中确认");
        }
        var job = jobs.find(nodeRunId, false);
        if (job == null || !job.canvasRunId().equals(runId)) {
            throw new BusinessException(404, "专业节点不存在");
        }
        var task = agents.findRun(request.getRunId());
        if (task == null || task.getCanvasScopeJson() == null || !Objects.equals(task.getUserId(), actor.userId())
                || !Objects.equals(task.getActorTeamId(), actor.teamId())
                || !nodeRunId.equals(json.read(task.getCanvasScopeJson(), CanvasAgentScope.class).nodeRunId())) {
            throw new BusinessException(403, "审批不属于此专业节点");
        }
        return confirmations.respond(request, actor.userId());
    }
}
