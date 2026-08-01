package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeModelFactory;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextFactory;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentScopeHarnessInvoker;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.ai.model.AiModelRequestOptions;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.PersistedAgentKernelSnapshotResolver;
import com.stonewu.fusion.service.ai.run.kernel.RunConfigUnavailableException;
import com.stonewu.fusion.service.ai.run.kernel.ToolManifestSnapshot;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.PendingConfirmation;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.AgentEvent;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.ToolUseBlock;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.time.Duration;
import java.util.List;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

@Component
public final class AgentExecutionFactory {

    private final AgentScopeHarnessInvoker harnessInvoker;
    private final AgentScopeRuntimeContextFactory runtimeContextFactory;
    private final AgentScopeEventMapper eventMapper;
    private final AiModelService modelService;
    private final AgentScopeModelFactory modelFactory;
    private final AgentRuntimeSchedulers schedulers;
    private final PersistedAgentKernelSnapshotResolver snapshotResolver;
    private final AiModelMetadataResolver modelMetadataResolver;
    private final ObjectMapper objectMapper;
    private final AgentKernelSpecFactory specFactory;
    private final Duration confirmationTimeout;

    public AgentExecutionFactory(
            AgentScopeHarnessInvoker harnessInvoker,
            AgentScopeRuntimeContextFactory runtimeContextFactory,
            AgentScopeEventMapper eventMapper,
            AiModelService modelService,
            AgentScopeModelFactory modelFactory,
            AgentRuntimeSchedulers schedulers,
            AiModelMetadataResolver modelMetadataResolver,
            ObjectMapper objectMapper,
            AgentKernelSpecFactory specFactory,
            AgentScopeV2Properties properties) {
        this.harnessInvoker = Objects.requireNonNull(harnessInvoker, "harnessInvoker must not be null");
        this.runtimeContextFactory = Objects.requireNonNull(
                runtimeContextFactory, "runtimeContextFactory must not be null");
        this.eventMapper = Objects.requireNonNull(eventMapper, "eventMapper must not be null");
        this.modelService = Objects.requireNonNull(modelService, "modelService must not be null");
        this.modelFactory = Objects.requireNonNull(modelFactory, "modelFactory must not be null");
        this.schedulers = Objects.requireNonNull(schedulers, "schedulers must not be null");
        this.modelMetadataResolver = Objects.requireNonNull(
                modelMetadataResolver, "modelMetadataResolver must not be null");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        this.specFactory = Objects.requireNonNull(specFactory, "specFactory must not be null");
        this.confirmationTimeout = Objects.requireNonNull(
                properties, "properties must not be null")
                .getExecution()
                .getConfirmationTimeout();
        this.snapshotResolver = new PersistedAgentKernelSnapshotResolver(this.objectMapper);
    }

    public Mono<AgentExecution> start(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            String stateSessionId,
            List<Msg> messages,
            AgentKernelSpec spec,
            AgentScopeRuntimeContextRequest runtimeRequest,
            Instant deadline) {
        return Mono.fromSupplier(() -> {
            requireRuntimeIdentity(
                    runId, ownerInstanceId, ownerEpoch,
                    stateSessionId, deadline, runtimeRequest);
            RuntimeContext runtimeContext = runtimeContextFactory.create(runtimeRequest);
            AgentRunContext run = runtimeRequest.run();
            return new AgentExecution(
                    runId,
                    ownerInstanceId,
                    ownerEpoch,
                    runtimeRequest.authenticatedUser().userId(),
                    stateSessionId,
                    runtimeRequest.parentRun(),
                    harnessInvoker.streamEvents(spec, List.copyOf(messages), runtimeContext)
                            .map(event -> mapConfirmationCandidate(
                                    event, run.deadline()))
                            .map(event -> childIdentity(event, run)),
                    ignored -> Mono.empty(),
                    () -> { });
        });
    }

    private AgentEventEnvelope mapConfirmationCandidate(
            AgentEvent event, Instant deadline) {
        AgentEventEnvelope mapped = eventMapper.map(event);
        if (!(event instanceof RequireUserConfirmEvent confirmation)) {
            return mapped;
        }
        PendingConfirmation candidate = confirmationCandidate(
                confirmation, mapped.payload());
        ObjectNode payload = mapped.payload().deepCopy();
        ObjectNode candidatePayload = JsonNodeFactory.instance.objectNode()
                .put("replyId", candidate.replyId())
                .put("pendingToolCallsJson", candidate.pendingToolCallsJson())
                .put("suspendedToolCallsJson", candidate.suspendedToolCallsJson())
                .put("expiresAt", candidate.expiresAt().toString());
        ArrayNode decisionIds = JsonNodeFactory.instance.arrayNode();
        candidate.decisionIds().stream().sorted().forEach(decisionIds::add);
        candidatePayload.set("decisionIds", decisionIds);
        payload.set("_platformConfirmationCandidate", candidatePayload);
        return new AgentEventEnvelope(
                mapped.rawEventId(),
                mapped.rawEventType(),
                mapped.source(),
                mapped.replyId(),
                mapped.blockId(),
                mapped.toolCallId(),
                mapped.parentToolCallId(),
                mapped.agentName(),
                mapped.outputType(),
                payload,
                mapped.createdAt());
    }

    private PendingConfirmation confirmationCandidate(
            RequireUserConfirmEvent event, JsonNode sanitizedPayload) {
        if (event.getReplyId() == null || event.getReplyId().isBlank()
                || event.getToolCalls().isEmpty()) {
            throw new IllegalStateException(
                    "AgentScope confirmation event has no actionable tool calls");
        }
        ArrayNode previews = JsonNodeFactory.instance.arrayNode();
        JsonNode sanitizedCalls = sanitizedPayload.path("toolCalls");
        if (!sanitizedCalls.isArray()
                || sanitizedCalls.size() != event.getToolCalls().size()) {
            throw new IllegalStateException(
                    "Sanitized AgentScope confirmation calls do not match the source event");
        }
        Set<String> decisionIds = new LinkedHashSet<>();
        for (int index = 0; index < event.getToolCalls().size(); index++) {
            ToolUseBlock toolCall = event.getToolCalls().get(index);
            if (toolCall.getId() == null || toolCall.getId().isBlank()
                    || toolCall.getName() == null || toolCall.getName().isBlank()
                    || !decisionIds.add(toolCall.getId())) {
                throw new IllegalStateException(
                        "AgentScope confirmation event contains invalid tool identity");
            }
            JsonNode sanitizedInput = sanitizedCalls.get(index).get("input");
            if (sanitizedInput == null) {
                throw new IllegalStateException(
                        "Sanitized AgentScope confirmation call has no input");
            }
            String argumentsPreview = writeJson(sanitizedInput);
            if (argumentsPreview.length() > 2048) {
                argumentsPreview = argumentsPreview.substring(0, 2048) + "…";
            }
            ObjectNode preview = JsonNodeFactory.instance.objectNode()
                    .put("toolCallId", toolCall.getId())
                    .put("toolName", toolCall.getName())
                    .put("argumentsPreview", argumentsPreview);
            previews.add(preview);
        }
        return new PendingConfirmation(
                event.getReplyId(),
                decisionIds,
                writeJson(previews),
                writeJson(suspendedToolCalls(event.getToolCalls())),
                Instant.now().plus(confirmationTimeout));
    }

    private ArrayNode suspendedToolCalls(List<ToolUseBlock> toolCalls) {
        ArrayNode suspended = JsonNodeFactory.instance.arrayNode();
        for (ToolUseBlock toolCall : toolCalls) {
            ObjectNode value = JsonNodeFactory.instance.objectNode()
                    .put("id", toolCall.getId())
                    .put("name", toolCall.getName())
                    .put("state", toolCall.getState().name());
            value.set("input", objectMapper.valueToTree(toolCall.getInput()));
            value.set("metadata", objectMapper.valueToTree(toolCall.getMetadata()));
            if (toolCall.getContent() != null) {
                value.put("content", toolCall.getContent());
            }
            suspended.add(value);
        }
        return suspended;
    }

    private String writeJson(Object value) {
        try {
            return objectMapper.writeValueAsString(value);
        } catch (JsonProcessingException failure) {
            throw new IllegalStateException(
                    "Failed to serialize AgentScope confirmation state", failure);
        }
    }

    private AgentEventEnvelope childIdentity(
            AgentEventEnvelope event, AgentRunContext run) {
        if (!run.childRun()) {
            return event;
        }
        return new AgentEventEnvelope(
                event.rawEventId(),
                event.rawEventType(),
                event.source(),
                event.replyId(),
                event.blockId(),
                event.toolCallId(),
                run.parentToolCallId(),
                run.agentName(),
                event.outputType(),
                event.payload(),
                event.createdAt());
    }

    /** Rehydrates only an exact, currently available no-tool kernel; other states fail closed. */
    public Mono<AgentKernelSpec> resolve(AgentKernelSnapshot snapshot) {
        AgentKernelSnapshot safeSnapshot = Objects.requireNonNull(
                snapshot, "snapshot must not be null");
        return Mono.fromCallable(() -> resolveBlocking(safeSnapshot))
                .subscribeOn(schedulers.modelBlocking());
    }

    private AgentKernelSpec resolveBlocking(AgentKernelSnapshot snapshot) {
        AgentKernelSnapshotPayload payload = snapshot.payload();
        long modelId;
        try {
            modelId = Long.parseLong(payload.modelConfigId());
        } catch (NumberFormatException invalidId) {
            throw unavailable("Persisted model configuration identity is invalid");
        }
        AiModel model = modelService.getById(modelId);
        if (model == null || !Integer.valueOf(1).equals(model.getStatus())) {
            throw unavailable("Persisted model configuration is unavailable");
        }
        String modelFingerprint = modelFactory.modelConfigFingerprint(model);
        long modelVersion = CanonicalAgentKernelSnapshotBuilder.modelConfigVersion(modelFingerprint);
        if (modelVersion != payload.modelConfigVersion()) {
            try {
                model = AiModelRequestOptions.withReasoningEffort(
                        model,
                        AiModelRequestOptions.reasoningEffort(payload.modelOptions()),
                        objectMapper);
            } catch (BusinessException invalidEffort) {
                throw unavailable("Persisted reasoning effort is no longer available");
            }
            modelFingerprint = modelFactory.modelConfigFingerprint(model);
            modelVersion = CanonicalAgentKernelSnapshotBuilder.modelConfigVersion(modelFingerprint);
        }
        AgentKernelSpec restored;
        try {
            restored = specFactory.restore(payload, model, modelFingerprint);
        } catch (RuntimeException unavailableTool) {
            throw unavailable(unavailableTool.getMessage());
        }
        List<ToolManifestSnapshot> currentTools = restored.toolManifest().stream()
                .map(tool -> new ToolManifestSnapshot(
                        tool.toolName(),
                        tool.schemaSha256(),
                        tool.readOnly(),
                        tool.concurrencySafe(),
                        restored.toolWhitelistVersion()))
                .toList();
        AgentKernelSnapshot validated = snapshotResolver.resolve(
                snapshot.snapshotJson(), snapshot.fingerprint(), modelVersion, currentTools);
        requireSameModel(validated.payload(), model);
        return restored;
    }

    private void requireSameModel(AgentKernelSnapshotPayload payload, AiModel model) {
        if (!Objects.equals(payload.modelCode(), model.getCode())
                || !payload.provider().equals(provider(model))) {
            throw unavailable("Persisted model configuration no longer matches the snapshot");
        }
    }

    private String provider(AiModel model) {
        String protocol = modelMetadataResolver.resolve(model).modelProtocol();
        return protocol == null ? "" : protocol.trim();
    }

    private void requireRuntimeIdentity(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            String stateSessionId,
            Instant deadline,
            AgentScopeRuntimeContextRequest request) {
        AgentRunContext run = Objects.requireNonNull(request, "runtimeRequest must not be null").run();
        if (!Objects.equals(runId, run.runId())
                || !Objects.equals(ownerInstanceId, run.ownerInstanceId())
                || ownerEpoch != run.ownerEpoch()
                || !Objects.equals(
                        stateSessionId,
                        request.conversation().agentStateSessionId())
                || !Objects.equals(deadline, run.deadline())) {
            throw new IllegalArgumentException(
                    "RuntimeContext run identity does not match durable execution ownership");
        }
        if (run.childRun()) {
            if (request.parentRun() == null
                    || !Objects.equals(
                            run.parentToolCallId(), request.parentRun().toolCallId())
                    || !Objects.equals(run.agentName(), request.parentRun().agentName())) {
                throw new IllegalArgumentException(
                        "Child RuntimeContext must carry its durable parent run identity");
            }
        } else if (request.parentRun() != null) {
            throw new IllegalArgumentException(
                    "Root RuntimeContext must not carry a parent run identity");
        }
    }

    private RunConfigUnavailableException unavailable(String message) {
        return new RunConfigUnavailableException(message);
    }
}
