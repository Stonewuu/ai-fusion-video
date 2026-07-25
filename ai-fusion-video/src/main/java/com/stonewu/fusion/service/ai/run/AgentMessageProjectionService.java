package com.stonewu.fusion.service.ai.run;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentEvent;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentEventMapper;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.AgentMessageService;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDateTime;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;

/** Rebuilds conversation messages deterministically from the committed event journal. */
@Service
@RequiredArgsConstructor
public class AgentMessageProjectionService {

    private static final int MAX_EVENTS_PER_TRANSACTION = 500;

    private final AgentRunMapper runMapper;
    private final AgentConversationMapper conversationMapper;
    private final AgentEventMapper eventMapper;
    private final AgentMessageMapper messageMapper;
    private final AgentMessageService messageService;
    private final ObjectMapper objectMapper;
    private final TransactionTemplate transactionTemplate;
    private final AgentRuntimeSchedulers schedulers;

    public Mono<Void> projectThrough(String runId, long throughSequence) {
        String safeRunId = requireRunId(runId);
        if (throughSequence < 0) {
            return Mono.error(new IllegalArgumentException(
                    "throughSequence must not be negative"));
        }
        return projectNextChunk(safeRunId, throughSequence);
    }

    public Mono<Void> projectCommitted(String runId) {
        String safeRunId = requireRunId(runId);
        return Mono.fromCallable(() -> runMapper.selectByRunId(safeRunId))
                .subscribeOn(schedulers.journal())
                .flatMap(run -> {
                    if (run == null) {
                        return Mono.error(new IllegalArgumentException(
                                "Agent run does not exist: " + safeRunId));
                    }
                    long lastCommitted = requireLastCommittedSequence(run);
                    return lastCommitted == 0
                            ? Mono.empty()
                            : projectThrough(safeRunId, lastCommitted);
                });
    }

    public Mono<Void> recoverTerminalBatch(int limit) {
        if (limit <= 0) {
            return Mono.error(new IllegalArgumentException("limit must be positive"));
        }
        return Mono.fromCallable(() ->
                        runMapper.selectProjectionRecoveryCandidates(limit))
                .subscribeOn(schedulers.journal())
                .flatMapMany(Flux::fromIterable)
                .flatMapDelayError(run -> projectThrough(
                                run.getRunId(), run.getTerminalSequence()),
                        1,
                        1)
                .then();
    }

    private Mono<Void> projectNextChunk(String runId, long throughSequence) {
        return Mono.fromCallable(() -> requireProgress(
                        transactionTemplate.execute(ignored ->
                                projectChunk(runId, throughSequence))))
                .subscribeOn(schedulers.journal())
                .flatMap(progress -> progress.cursor() >= throughSequence
                        ? Mono.empty()
                        : projectNextChunk(runId, throughSequence));
    }

    private ProjectionProgress projectChunk(String runId, long throughSequence) {
        AgentRun run = runMapper.selectByRunIdForUpdate(runId);
        if (run == null) {
            throw new IllegalArgumentException("Agent run does not exist: " + runId);
        }
        long cursor = requireCursor(run);
        long lastCommitted = requireLastCommittedSequence(run);
        if (cursor > lastCommitted) {
            throw new IllegalStateException(
                    "Agent projection cursor exceeds committed history: "
                            + cursor + " > " + lastCommitted);
        }
        if (throughSequence > lastCommitted) {
            throw new IllegalArgumentException(
                    "Projection target exceeds the last committed sequence: "
                            + throughSequence + " > " + lastCommitted);
        }
        if (cursor >= throughSequence) {
            repairProjectionCompletion(run, cursor);
            return new ProjectionProgress(cursor);
        }

        List<AgentEvent> events = eventMapper.selectProjectionRange(
                runId,
                cursor,
                throughSequence,
                MAX_EVENTS_PER_TRANSACTION);
        if (events.isEmpty()) {
            throw new IllegalStateException(
                    "Committed Agent event range contains a gap after sequence " + cursor);
        }

        long expected = cursor + 1;
        for (AgentEvent event : events) {
            if (!Objects.equals(event.getSequenceNo(), expected)) {
                throw new IllegalStateException(
                        "Committed Agent event sequence gap: expected " + expected
                                + " but found " + event.getSequenceNo());
            }
            projectBoundary(run, event);
            expected++;
        }
        long projectedThrough = events.getLast().getSequenceNo();
        advanceCursor(run, projectedThrough);
        return new ProjectionProgress(projectedThrough);
    }

    private void projectBoundary(AgentRun run, AgentEvent event) {
        if (payload(event).path("_platformMirroredChildEvent").asBoolean(false)) {
            return;
        }
        String outputType = event.getOutputType();
        if (outputType == null
                || "CONTENT".equals(outputType)
                || "REASONING".equals(outputType)
                || "SUB_AGENT_STARTED".equals(outputType)) {
            return;
        }

        projectAssistantSegment(run, event);
        if ("TOOL_CALL".equals(outputType)) {
            persistProjection(run, event, "tool-call", toolCallProjection(run, event));
        } else if ("TOOL_FINISHED".equals(outputType)) {
            persistProjection(run, event, "tool-result", toolResultProjection(run, event));
        }
    }

    private void projectAssistantSegment(AgentRun run, AgentEvent boundary) {
        long previousBoundary = eventMapper.selectLastContextBoundarySequence(
                run.getRunId(),
                boundary.getSequenceNo(),
                boundary.getParentToolCallId());
        List<AgentEvent> deltas = eventMapper.selectContextDeltas(
                run.getRunId(),
                previousBoundary,
                boundary.getSequenceNo(),
                boundary.getParentToolCallId());
        StringBuilder content = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        Long reasoningDurationMs = nonNegativeLong(
                payload(boundary), "reasoningDurationMs");
        for (AgentEvent delta : deltas) {
            JsonNode deltaPayload = payload(delta);
            String value = deltaValue(deltaPayload);
            if (value == null) {
                continue;
            }
            if ("CONTENT".equals(delta.getOutputType())) {
                content.append(value);
                Long deltaDurationMs = nonNegativeLong(
                        deltaPayload, "reasoningDurationMs");
                if (deltaDurationMs != null) {
                    reasoningDurationMs = deltaDurationMs;
                }
            } else if ("REASONING".equals(delta.getOutputType())) {
                reasoning.append(value);
            }
        }
        if (content.isEmpty() && reasoning.isEmpty()) {
            return;
        }
        AgentMessage message = AgentMessage.builder()
                .conversationId(run.getConversationId())
                .runId(run.getRunId())
                .role("assistant")
                .content(content.isEmpty() ? null : content.toString())
                .reasoningContent(reasoning.isEmpty() ? null : reasoning.toString())
                .reasoningDurationMs(reasoning.isEmpty() ? null : reasoningDurationMs)
                .parentToolCallId(projectedParentToolCallId(run, boundary))
                .build();
        persistProjection(run, boundary, "assistant", message);
    }

    private AgentMessage toolCallProjection(AgentRun run, AgentEvent event) {
        JsonNode payload = payload(event);
        String toolName = firstText(payload, "toolCallName", "toolName", "name");
        String arguments = collectToolDeltas(
                run.getRunId(), event.getToolCallId(), event.getSequenceNo(), true);
        if (arguments == null) {
            arguments = firstText(payload, "arguments", "input", "delta");
        }
        return AgentMessage.builder()
                .conversationId(run.getConversationId())
                .runId(run.getRunId())
                .role("tool")
                .content(arguments)
                .toolName(requireText(toolName, "toolName"))
                .toolStatus("running")
                .toolCallId(requireText(event.getToolCallId(), "toolCallId"))
                .parentToolCallId(projectedParentToolCallId(run, event))
                .build();
    }

    private AgentMessage toolResultProjection(AgentRun run, AgentEvent event) {
        JsonNode payload = payload(event);
        String toolName = firstText(payload, "toolCallName", "toolName", "name");
        String result = collectToolDeltas(
                run.getRunId(), event.getToolCallId(), event.getSequenceNo(), false);
        if (result == null) {
            result = firstText(payload, "result", "output", "delta");
        }
        String state = firstText(payload, "state", "toolStatus", "status");
        String status = AgentScopeToolResultStatus.project(state, result);
        return AgentMessage.builder()
                .conversationId(run.getConversationId())
                .runId(run.getRunId())
                .role("tool")
                .content(result)
                .toolName(requireText(toolName, "toolName"))
                .toolStatus(status)
                .toolCallId(requireText(event.getToolCallId(), "toolCallId"))
                .parentToolCallId(projectedParentToolCallId(run, event))
                .build();
    }

    private String projectedParentToolCallId(AgentRun run, AgentEvent event) {
        return event.getParentToolCallId() != null
                ? event.getParentToolCallId()
                : run.getParentToolCallId();
    }

    private String collectToolDeltas(
            String runId,
            String toolCallId,
            long throughSequence,
            boolean callArguments) {
        String safeToolCallId = requireText(toolCallId, "toolCallId");
        StringBuilder value = new StringBuilder();
        for (AgentEvent delta : eventMapper.selectToolDeltas(
                runId, safeToolCallId, throughSequence)) {
            boolean matches = callArguments
                    ? "TOOL_CALL_DELTA".equals(delta.getRawEventType())
                    : delta.getRawEventType().startsWith("TOOL_RESULT_");
            if (!matches) {
                continue;
            }
            JsonNode deltaNode = payload(delta).get("delta");
            if (deltaNode == null || deltaNode.isNull()) {
                continue;
            }
            if (deltaNode.isTextual()) {
                value.append(deltaNode.textValue());
            } else {
                try {
                    value.append(objectMapper.writeValueAsString(deltaNode));
                } catch (JsonProcessingException serializationFailure) {
                    throw new IllegalStateException(
                            "Tool delta projection serialization failed",
                            serializationFailure);
                }
            }
        }
        return value.isEmpty() ? null : value.toString();
    }

    private void persistProjection(
            AgentRun run,
            AgentEvent boundary,
            String kind,
            AgentMessage candidate) {
        String projectionKey = projectionKey(
                run.getRunId(), boundary.getSequenceNo(), kind);
        candidate.setConversationId(run.getConversationId());
        candidate.setRunId(run.getRunId());
        candidate.setProjectionKey(projectionKey);
        AgentMessage existing = messageMapper.selectByProjectionKey(projectionKey);
        if (existing != null) {
            requireSameProjection(existing, candidate);
            return;
        }
        messageService.saveProjectedMessage(run.getConversationId(), candidate);
    }

    private void requireSameProjection(AgentMessage existing, AgentMessage candidate) {
        if (!Objects.equals(existing.getConversationId(), candidate.getConversationId())
                || !Objects.equals(existing.getRunId(), candidate.getRunId())
                || !Objects.equals(existing.getProjectionKey(), candidate.getProjectionKey())
                || !Objects.equals(existing.getRole(), candidate.getRole())
                || !Objects.equals(existing.getContent(), candidate.getContent())
                || !Objects.equals(existing.getReferencesJson(), candidate.getReferencesJson())
                || !Objects.equals(existing.getToolName(), candidate.getToolName())
                || !Objects.equals(existing.getToolStatus(), candidate.getToolStatus())
                || !Objects.equals(existing.getToolCallId(), candidate.getToolCallId())
                || !Objects.equals(existing.getParentToolCallId(),
                        candidate.getParentToolCallId())
                || !Objects.equals(existing.getReasoningContent(),
                        candidate.getReasoningContent())
                || !Objects.equals(existing.getReasoningDurationMs(),
                        candidate.getReasoningDurationMs())) {
            throw new IllegalStateException(
                    "Projection key resolves to a different Agent message: "
                            + candidate.getProjectionKey());
        }
    }

    private void advanceCursor(AgentRun run, long projectedThrough) {
        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        boolean complete = run.getTerminalSequence() != null
                && projectedThrough >= run.getTerminalSequence();
        if (complete) {
            finishTerminalProjection(run, databaseNow);
        }
        int updated = runMapper.update(null, new LambdaUpdateWrapper<AgentRun>()
                .eq(AgentRun::getId, run.getId())
                .eq(AgentRun::getProjectedThroughSequence,
                        run.getProjectedThroughSequence())
                .set(AgentRun::getProjectedThroughSequence, projectedThrough)
                .set(AgentRun::getProjectionCompletedAt,
                        complete ? databaseNow : run.getProjectionCompletedAt())
                .set(AgentRun::getUpdateTime, databaseNow));
        if (updated != 1) {
            throw new IllegalStateException(
                    "Agent projection cursor update did not affect exactly one row");
        }
    }

    private void repairProjectionCompletion(AgentRun run, long cursor) {
        if (run.getTerminalSequence() == null
                || cursor < run.getTerminalSequence()) {
            return;
        }
        LocalDateTime databaseNow = runMapper.selectDatabaseNow();
        finishTerminalProjection(run, databaseNow);
        if (run.getProjectionCompletedAt() != null) {
            return;
        }
        if (runMapper.update(null, new LambdaUpdateWrapper<AgentRun>()
                .eq(AgentRun::getId, run.getId())
                .isNull(AgentRun::getProjectionCompletedAt)
                .set(AgentRun::getProjectionCompletedAt, databaseNow)
                .set(AgentRun::getUpdateTime, databaseNow)) != 1) {
            throw new IllegalStateException(
                    "Agent projection completion repair did not affect exactly one row");
        }
    }

    private void finishTerminalProjection(AgentRun run, LocalDateTime databaseNow) {
        String terminalOutputType = requireText(
                run.getTerminalOutputType(), "terminalOutputType");
        if ("CANCELLED".equals(terminalOutputType)) {
            messageMapper.cancelUnfinishedTools(run.getRunId(), databaseNow);
        }
        if (run.getParentRunId() != null) {
            return;
        }
        String status = switch (terminalOutputType) {
            case "DONE" -> "completed";
            case "ERROR" -> "error";
            case "CANCELLED" -> "cancelled";
            default -> throw new IllegalStateException(
                            "Unsupported Agent terminal output type: "
                            + run.getTerminalOutputType());
        };
        if (conversationMapper.update(null,
                new LambdaUpdateWrapper<AgentConversation>()
                        .eq(AgentConversation::getConversationId,
                                run.getConversationId())
                        .eq(AgentConversation::getDeleted, false)
                        .set(AgentConversation::getStatus, status)
                        .set(AgentConversation::getUpdateTime, databaseNow)) != 1) {
            throw new IllegalStateException(
                    "Agent conversation terminal update did not affect exactly one row");
        }
    }

    private JsonNode payload(AgentEvent event) {
        try {
            JsonNode payload = objectMapper.readTree(event.getPayloadJson());
            if (payload == null || !payload.isObject()) {
                throw new IllegalStateException(
                        "Agent event payload must be a JSON object: " + event.getId());
            }
            return payload;
        } catch (JsonProcessingException invalid) {
            throw new IllegalStateException(
                    "Agent event payload is invalid JSON: " + event.getId(), invalid);
        }
    }

    private String deltaValue(JsonNode payload) {
        return firstText(payload, "delta", "content", "reasoningContent");
    }

    private Long nonNegativeLong(JsonNode payload, String field) {
        JsonNode value = payload.get(field);
        if (value == null || value.isNull()) {
            return null;
        }
        if (!value.canConvertToLong() || value.longValue() < 0) {
            throw new IllegalStateException(
                    "Projected Agent event field must be a non-negative integer: " + field);
        }
        return value.longValue();
    }

    private String firstText(JsonNode payload, String... fields) {
        for (String field : fields) {
            JsonNode value = payload.get(field);
            if (value == null || value.isNull()) {
                continue;
            }
            if (!value.isTextual()) {
                throw new IllegalStateException(
                        "Projected Agent event field must be text: " + field);
            }
            return value.textValue();
        }
        return null;
    }

    static String projectionKey(String runId, long sequence, String kind) {
        String material = runId + ':' + sequence + ':' + kind;
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(material.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }

    private long requireCursor(AgentRun run) {
        Long cursor = run.getProjectedThroughSequence();
        if (cursor == null || cursor < 0) {
            throw new IllegalStateException(
                    "Agent run has an invalid projection cursor: " + run.getRunId());
        }
        return cursor;
    }

    private long requireLastCommittedSequence(AgentRun run) {
        Long nextSequence = run.getNextSequence();
        if (nextSequence == null || nextSequence < 1) {
            throw new IllegalStateException(
                    "Agent run has an invalid next sequence: " + run.getRunId());
        }
        return nextSequence - 1;
    }

    private String requireRunId(String runId) {
        String value = requireText(runId, "runId");
        if (value.length() > 64
                || !StandardCharsets.US_ASCII.newEncoder().canEncode(value)) {
            throw new IllegalArgumentException(
                    "runId must be at most 64 ASCII characters");
        }
        return value;
    }

    private String requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return value;
    }

    private ProjectionProgress requireProgress(ProjectionProgress progress) {
        return Objects.requireNonNull(
                progress, "Agent projection transaction returned no result");
    }

    private record ProjectionProgress(long cursor) {
    }
}
