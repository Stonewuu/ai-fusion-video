package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.enums.ai.AgentTerminalOutputType;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreSlot;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.RunTerminalRequest;
import com.stonewu.fusion.service.ai.run.model.SystemTerminalActor;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;

/** Ends an expired confirmation when application code observes its deadline. */
@Component
public final class AgentConfirmationExpiryCoordinator {

    public static final String REASON = "CONFIRMATION_EXPIRED";
    public static final String MESSAGE = "审批时间已结束，相关操作未执行。";

    private final AgentRunMapper runMapper;
    private final AgentEventMapper eventMapper;
    private final ObjectMapper objectMapper;
    private final RunTerminalCoordinator terminals;
    private final AgentMessageProjectionService projections;
    private final AgentEventEnvelopeSanitizer sanitizer;
    private final AgentRuntimeSchedulers schedulers;

    public AgentConfirmationExpiryCoordinator(
            AgentRunMapper runMapper,
            AgentEventMapper eventMapper,
            ObjectMapper objectMapper,
            RunTerminalCoordinator terminals,
            AgentMessageProjectionService projections,
            AgentEventEnvelopeSanitizer sanitizer,
            AgentRuntimeSchedulers schedulers) {
        this.runMapper = Objects.requireNonNull(runMapper, "runMapper must not be null");
        this.eventMapper = Objects.requireNonNull(
                eventMapper, "eventMapper must not be null");
        this.objectMapper = Objects.requireNonNull(
                objectMapper, "objectMapper must not be null");
        this.terminals = Objects.requireNonNull(terminals, "terminals must not be null");
        this.projections = Objects.requireNonNull(
                projections, "projections must not be null");
        this.sanitizer = Objects.requireNonNull(sanitizer, "sanitizer must not be null");
        this.schedulers = Objects.requireNonNull(
                schedulers, "schedulers must not be null");
    }

    public Mono<Boolean> expireAuthorized(
            String runId, String replyId, long currentUserId) {
        String safeRunId = requireText(runId, "runId");
        String safeReplyId = requireText(replyId, "replyId");
        if (currentUserId <= 0) {
            return Mono.error(new IllegalArgumentException(
                    "currentUserId must be positive"));
        }
        return journal(() -> {
                    AgentRun run = runMapper.selectAuthorizedByRunId(
                            safeRunId, currentUserId);
                    if (run == null) {
                        throw new BusinessException(404, "Agent 运行不存在");
                    }
                    if (AgentRunStatus.WAITING_CONFIRMATION.name().equals(run.getStatus())
                            && !safeReplyId.equals(run.getWaitingReplyId())) {
                        throw new BusinessException(409, "该审批已处理或已失效");
                    }
                    return run;
                })
                .flatMap(this::expireIfNeeded);
    }

    public Mono<Boolean> expireIfNeeded(AgentRun run) {
        AgentRun safeRun = Objects.requireNonNull(run, "run must not be null");
        if (!AgentRunStatus.WAITING_CONFIRMATION.name().equals(safeRun.getStatus())
                || !expired(safeRun)) {
            return Mono.just(false);
        }
        return terminalRequest(safeRun)
                .flatMap(request -> terminals.terminateSystem(
                        request, SystemTerminalActor.CONFIRMATION_EXPIRER))
                .flatMap(committed -> committed
                        .map(event -> projections.projectThrough(
                                        event.runId(), event.sequence())
                                .thenReturn(true))
                        .orElseGet(() -> Mono.just(false)));
    }

    private Mono<RunTerminalRequest> terminalRequest(AgentRun run) {
        return journal(() -> {
            String replyId = requireText(run.getWaitingReplyId(), "waitingReplyId");
            AgentEvent candidate = eventMapper.selectConfirmationCandidate(
                    run.getRunId(), replyId);
            if (candidate == null) {
                throw new IllegalStateException(
                        "Expired confirmation has no durable candidate");
            }
            ArrayNode pendingToolCalls = pendingToolCalls(candidate);
            ObjectNode payload = JsonNodeFactory.instance.objectNode()
                    .put("outputType", AgentTerminalOutputType.CANCELLED.name())
                    .put("cancellationReason", REASON)
                    .put("content", MESSAGE)
                    .put("finished", true);
            payload.set("pendingToolCalls", pendingToolCalls);
            AgentEventEnvelope envelope = new AgentEventEnvelope(
                    "confirmation-expired-"
                            + UUID.randomUUID().toString().replace("-", ""),
                    REASON,
                    run.getParentRunId() == null
                            ? "main"
                            : "main/" + run.getAgentName(),
                    replyId,
                    null,
                    null,
                    run.getParentToolCallId(),
                    run.getAgentName(),
                    AgentTerminalOutputType.CANCELLED.name(),
                    sanitizer.sanitize(payload),
                    Instant.now());
            return new RunTerminalRequest(
                    run.getRunId(),
                    new StateStoreSlot(
                            String.valueOf(run.getUserId()),
                            run.getAgentStateSessionId()),
                    Set.of(AgentRunStatus.WAITING_CONFIRMATION),
                    AgentRunStatus.CANCELLED,
                    AgentTerminalOutputType.CANCELLED,
                    null,
                    null,
                    envelope);
        });
    }

    private ArrayNode pendingToolCalls(AgentEvent event) {
        try {
            JsonNode payload = objectMapper.readTree(event.getPayloadJson());
            JsonNode encoded = payload == null ? null : payload.get("pendingToolCallsJson");
            if (encoded == null || !encoded.isTextual()) {
                throw new IllegalStateException(
                        "Confirmation candidate has no pending tool calls");
            }
            JsonNode parsed = objectMapper.readTree(encoded.textValue());
            if (!(parsed instanceof ArrayNode array) || array.isEmpty()) {
                throw new IllegalStateException(
                        "Confirmation candidate pending tool calls are invalid");
            }
            return array;
        } catch (JsonProcessingException invalidJson) {
            throw new IllegalStateException(
                    "Confirmation candidate pending tool calls are invalid", invalidJson);
        }
    }

    private boolean expired(AgentRun run) {
        LocalDateTime expiresAt = run.getWaitExpiresAt();
        return expiresAt != null
                && !expiresAt.toInstant(ZoneOffset.UTC).isAfter(Instant.now());
    }

    private String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new BusinessException(400, field + " 不能为空");
        }
        return value.trim();
    }

    private <T> Mono<T> journal(java.util.concurrent.Callable<T> operation) {
        return Mono.fromCallable(operation)
                .subscribeOn(schedulers.journal());
    }
}
