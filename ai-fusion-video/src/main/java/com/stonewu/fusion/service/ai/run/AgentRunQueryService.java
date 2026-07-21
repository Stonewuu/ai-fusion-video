package com.stonewu.fusion.service.ai.run;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.controller.ai.vo.PipelineRunStatusRespVO;
import com.stonewu.fusion.controller.ai.vo.RunningPipelineRunRespVO;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.repository.ai.AgentEventRepository;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Supplier;

/** Read-only authorization and projection boundary for durable pipeline APIs. */
@Service
public final class AgentRunQueryService {

    private static final Set<String> TERMINAL_OUTPUTS =
            Set.of("DONE", "ERROR", "CANCELLED");

    private final AgentRunMapper runMapper;
    private final AgentConversationMapper conversationMapper;
    private final AgentEventMapper eventMapper;
    private final AgentEventRepository eventRepository;
    private final ObjectMapper objectMapper;
    private final AgentRuntimeSchedulers schedulers;

    public AgentRunQueryService(
            AgentRunMapper runMapper,
            AgentConversationMapper conversationMapper,
            AgentEventMapper eventMapper,
            AgentEventRepository eventRepository,
            ObjectMapper objectMapper,
            AgentRuntimeSchedulers schedulers) {
        this.runMapper = Objects.requireNonNull(runMapper, "runMapper must not be null");
        this.conversationMapper = Objects.requireNonNull(
                conversationMapper, "conversationMapper must not be null");
        this.eventMapper = Objects.requireNonNull(
                eventMapper, "eventMapper must not be null");
        this.eventRepository = Objects.requireNonNull(
                eventRepository, "eventRepository must not be null");
        this.objectMapper = Objects.requireNonNull(
                objectMapper, "objectMapper must not be null");
        this.schedulers = Objects.requireNonNull(
                schedulers, "schedulers must not be null");
    }

    public Mono<Void> authorizeConversationForStart(
            String conversationId, long currentUserId) {
        requireUserId(currentUserId);
        if (conversationId == null || conversationId.isBlank()) {
            return Mono.empty();
        }
        String safeConversationId = requireIdentifier(
                conversationId, "conversationId");
        return journal(() -> {
                    AgentConversation conversation = conversationMapper.selectOne(
                            new LambdaQueryWrapper<AgentConversation>()
                                    .eq(AgentConversation::getConversationId,
                                            safeConversationId)
                                    .eq(AgentConversation::getDeleted, false));
                    if (conversation != null
                            && !Objects.equals(
                                    conversation.getUserId(), currentUserId)) {
                        throw notFound();
                    }
                    return Boolean.TRUE;
                })
                .then();
    }

    public Mono<AgentRun> requireAuthorizedRun(
            String runId, long currentUserId) {
        String safeRunId = requireIdentifier(runId, "runId");
        requireUserId(currentUserId);
        return journal(() -> requireAuthorizedRunNow(
                safeRunId, currentUserId));
    }

    public Mono<AgentRun> resolveAuthorizedTarget(
            String runId,
            String conversationId,
            long currentUserId) {
        requireUserId(currentUserId);
        String safeRunId = optionalIdentifier(runId, "runId");
        String safeConversationId = optionalIdentifier(
                conversationId, "conversationId");
        if (safeRunId == null && safeConversationId == null) {
            return Mono.error(new BusinessException(
                    400, "runId or conversationId is required"));
        }
        return journal(() -> {
            AgentRun run;
            if (safeRunId != null) {
                run = requireAuthorizedRunNow(safeRunId, currentUserId);
                if (safeConversationId != null
                        && !safeConversationId.equals(run.getConversationId())) {
                    throw new BusinessException(
                            400, "conversationId does not match runId");
                }
            } else {
                run = runMapper.selectAuthorizedPreferredRootByConversation(
                        safeConversationId, currentUserId);
                if (run == null) {
                    throw notFound();
                }
            }
            return run;
        });
    }

    public Mono<PipelineRunStatusRespVO> status(
            String runId,
            String conversationId,
            long currentUserId) {
        return resolveAuthorizedTarget(runId, conversationId, currentUserId)
                .flatMap(run -> terminalEvent(run)
                        .map(terminal -> status(run, terminal))
                        .defaultIfEmpty(status(run, null)));
    }

    public Mono<List<RunningPipelineRunRespVO>> listRunning(
            long currentUserId) {
        requireUserId(currentUserId);
        return journal(() -> runningNow(currentUserId));
    }

    public Mono<AiChatStreamRespVO> project(
            AgentRun run, CommittedAgentEvent event) {
        Objects.requireNonNull(run, "run must not be null");
        Objects.requireNonNull(event, "event must not be null");
        if (!Objects.equals(run.getRunId(), event.runId())) {
            return Mono.error(new IllegalArgumentException(
                    "event runId does not match authorized run"));
        }
        if ("TOOL_CALL".equals(event.outputType())
                || "TOOL_FINISHED".equals(event.outputType())) {
            return journal(() -> projectNow(run, event));
        }
        return Mono.fromSupplier(() -> projectNow(run, event));
    }

    private Mono<AiChatStreamRespVO> terminalEvent(AgentRun run) {
        Long terminalSequence = run.getTerminalSequence();
        if (terminalSequence == null) {
            return Mono.empty();
        }
        return journal(() -> {
            List<CommittedAgentEvent> events = eventRepository.loadReplayPage(
                    run.getRunId(), terminalSequence - 1, terminalSequence, 1);
            if (events.size() != 1
                    || events.getFirst().sequence() != terminalSequence
                    || !events.getFirst().publishRequired()) {
                throw new IllegalStateException(
                        "Agent terminal event is not replayable: " + run.getRunId());
            }
            return projectNow(run, events.getFirst());
        });
    }

    private PipelineRunStatusRespVO status(
            AgentRun run, AiChatStreamRespVO terminalEvent) {
        return new PipelineRunStatusRespVO(
                run.getRunId(),
                run.getStatus(),
                lastSequence(run),
                run.getWaitingReplyId(),
                terminalEvent);
    }

    private List<RunningPipelineRunRespVO> runningNow(long currentUserId) {
        List<AgentRun> runs = runMapper.selectAuthorizedRunningRoots(currentUserId);
        if (runs.isEmpty()) {
            return List.of();
        }
        LinkedHashSet<String> conversationIds = new LinkedHashSet<>();
        for (AgentRun run : runs) {
            conversationIds.add(run.getConversationId());
        }
        List<AgentConversation> conversations = conversationMapper.selectList(
                new LambdaQueryWrapper<AgentConversation>()
                        .in(AgentConversation::getConversationId, conversationIds)
                        .eq(AgentConversation::getUserId, currentUserId)
                        .eq(AgentConversation::getDeleted, false));
        Map<String, AgentConversation> byId = new LinkedHashMap<>();
        for (AgentConversation conversation : conversations) {
            byId.put(conversation.getConversationId(), conversation);
        }
        List<RunningPipelineRunRespVO> result = new ArrayList<>(runs.size());
        for (AgentRun run : runs) {
            AgentConversation conversation = byId.get(run.getConversationId());
            if (conversation == null) {
                throw new IllegalStateException(
                        "Authorized Agent run lost its conversation: "
                                + run.getRunId());
            }
            result.add(new RunningPipelineRunRespVO(
                    run.getRunId(),
                    run.getConversationId(),
                    run.getProjectId(),
                    conversation.getTitle(),
                    conversation.getCategory(),
                    run.getStatus(),
                    lastSequence(run),
                    run.getWaitingReplyId(),
                    instant(run.getStartedAt(), "startedAt", run.getRunId())));
        }
        return List.copyOf(result);
    }

    private AgentRun requireAuthorizedRunNow(
            String runId, long currentUserId) {
        AgentRun run = runMapper.selectAuthorizedByRunId(runId, currentUserId);
        if (run == null) {
            throw notFound();
        }
        return run;
    }

    private AiChatStreamRespVO projectNow(
            AgentRun run, CommittedAgentEvent event) {
        JsonNode payload = event.projection();
        if (payload == null || !payload.isObject()) {
            throw new IllegalStateException(
                    "Published Agent event has no object projection: "
                            + event.eventId());
        }
        AiChatStreamRespVO response = objectMapper.convertValue(
                payload, AiChatStreamRespVO.class);
        AgentEventEnvelopeView identity = new AgentEventEnvelopeView(event);
        response.setSchemaVersion(1)
                .setRunId(event.runId())
                .setSequence(event.sequence())
                .setMessageId(response.getMessageId() == null
                        ? identity.rawEventId()
                        : response.getMessageId())
                .setConversationId(run.getConversationId())
                .setOutputType(event.outputType())
                .setSource(identity.source())
                .setReplyId(identity.replyId())
                .setBlockId(identity.blockId())
                .setToolCallId(identity.toolCallId())
                .setParentToolCallId(identity.parentToolCallId())
                .setAgentName(identity.agentName())
                .setRawEventId(identity.rawEventId())
                .setRawEventType(identity.rawEventType())
                .setCreatedAt(identity.createdAt());

        switch (event.outputType()) {
            case "CONTENT" -> response.setContent(firstText(
                    payload, "content", "delta"));
            case "REASONING" -> response.setReasoningContent(firstText(
                    payload, "reasoningContent", "delta"));
            case "TOOL_CALL" -> projectToolCall(response, event, payload);
            case "TOOL_FINISHED" -> projectToolResult(response, event, payload);
            default -> {
                // Other stable projections already use their persisted payload.
            }
        }
        if (TERMINAL_OUTPUTS.contains(event.outputType())) {
            response.setFinished(true);
        } else if (response.getFinished() == null) {
            response.setFinished(false);
        }
        return response;
    }

    private void projectToolCall(
            AiChatStreamRespVO response,
            CommittedAgentEvent event,
            JsonNode payload) {
        String toolCallId = requireText(
                event.envelope().toolCallId(), "toolCallId");
        String toolName = requireText(firstText(
                payload, "toolCallName", "toolName", "name"), "toolName");
        String arguments = collectToolDeltas(
                event.runId(), toolCallId, event.sequence(), true);
        if (arguments == null) {
            arguments = firstValue(payload, "arguments", "input", "delta");
        }
        response.setToolCallId(toolCallId)
                .setToolName(toolName)
                .setToolCalls(List.of(new AiChatStreamRespVO.ToolCallVO()
                        .setId(toolCallId)
                        .setName(toolName)
                        .setArguments(arguments)));
    }

    private void projectToolResult(
            AiChatStreamRespVO response,
            CommittedAgentEvent event,
            JsonNode payload) {
        String toolCallId = requireText(
                event.envelope().toolCallId(), "toolCallId");
        String toolName = requireText(firstText(
                payload, "toolCallName", "toolName", "name"), "toolName");
        String result = collectToolDeltas(
                event.runId(), toolCallId, event.sequence(), false);
        if (result == null) {
            result = firstValue(payload, "result", "output", "delta");
        }
        String state = firstText(payload, "state", "toolStatus", "status");
        String status = state != null
                && ("ERROR".equalsIgnoreCase(state)
                    || "FAILED".equalsIgnoreCase(state))
                ? "error"
                : "success";
        response.setToolCallId(toolCallId)
                .setToolName(toolName)
                .setToolResult(result)
                .setToolStatus(status);
    }

    private String collectToolDeltas(
            String runId,
            String toolCallId,
            long throughSequence,
            boolean arguments) {
        StringBuilder collected = new StringBuilder();
        for (AgentEvent row : eventMapper.selectToolDeltas(
                runId, toolCallId, throughSequence)) {
            boolean matches = arguments
                    ? "TOOL_CALL_DELTA".equals(row.getRawEventType())
                    : row.getRawEventType().startsWith("TOOL_RESULT_");
            if (!matches) {
                continue;
            }
            JsonNode delta = eventPayload(row).get("delta");
            if (delta == null || delta.isNull()) {
                continue;
            }
            collected.append(delta.isTextual()
                    ? delta.textValue()
                    : json(delta, "tool delta"));
        }
        return collected.isEmpty() ? null : collected.toString();
    }

    private JsonNode eventPayload(AgentEvent event) {
        if (event.getPayloadJson() == null) {
            throw new IllegalStateException(
                    "Agent event payload is missing: " + event.getId());
        }
        try {
            JsonNode payload = objectMapper.readTree(event.getPayloadJson());
            if (payload == null || !payload.isObject()) {
                throw new IllegalStateException(
                        "Agent event payload is not an object: " + event.getId());
            }
            return payload;
        } catch (JsonProcessingException invalid) {
            throw new IllegalStateException(
                    "Agent event payload is invalid: " + event.getId(), invalid);
        }
    }

    private String firstText(JsonNode payload, String... fields) {
        for (String field : fields) {
            JsonNode value = payload.get(field);
            if (value != null && value.isTextual() && !value.textValue().isBlank()) {
                return value.textValue();
            }
        }
        return null;
    }

    private String firstValue(JsonNode payload, String... fields) {
        for (String field : fields) {
            JsonNode value = payload.get(field);
            if (value == null || value.isNull()) {
                continue;
            }
            return value.isTextual()
                    ? value.textValue()
                    : json(value, field);
        }
        return null;
    }

    private String json(JsonNode value, String field) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException(
                    "Agent " + field + " serialization failed", failure);
        }
    }

    private long lastSequence(AgentRun run) {
        Long nextSequence = run.getNextSequence();
        if (nextSequence == null || nextSequence < 1) {
            throw new IllegalStateException(
                    "Agent run has an invalid next sequence: " + run.getRunId());
        }
        return nextSequence - 1;
    }

    private Instant instant(
            LocalDateTime value, String field, String runId) {
        if (value == null) {
            throw new IllegalStateException(
                    "Agent run " + field + " is missing: " + runId);
        }
        return value.toInstant(ZoneOffset.UTC);
    }

    private String optionalIdentifier(String value, String field) {
        return value == null ? null : requireIdentifier(value, field);
    }

    private String requireIdentifier(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new BusinessException(400, field + " must not be blank");
        }
        if (value.length() > 64
                || !StandardCharsets.US_ASCII.newEncoder().canEncode(value)) {
            throw new BusinessException(
                    400, field + " must be at most 64 ASCII characters");
        }
        return value;
    }

    private String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalStateException(field + " must not be blank");
        }
        return value;
    }

    private void requireUserId(long currentUserId) {
        if (currentUserId <= 0) {
            throw new IllegalArgumentException(
                    "currentUserId must be positive");
        }
    }

    private BusinessException notFound() {
        return new BusinessException(404, "Agent run does not exist");
    }

    private <T> Mono<T> journal(Supplier<T> operation) {
        return Mono.fromCallable(operation::get)
                .subscribeOn(schedulers.journal());
    }

    private record AgentEventEnvelopeView(
            String source,
            String replyId,
            String blockId,
            String toolCallId,
            String parentToolCallId,
            String agentName,
            String rawEventId,
            String rawEventType,
            Instant createdAt) {

        private AgentEventEnvelopeView(CommittedAgentEvent event) {
            this(
                    event.envelope().source(),
                    event.envelope().replyId(),
                    event.envelope().blockId(),
                    event.envelope().toolCallId(),
                    event.envelope().parentToolCallId(),
                    event.envelope().agentName(),
                    event.envelope().rawEventId(),
                    event.envelope().rawEventType(),
                    event.envelope().createdAt());
        }
    }
}
