package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeModelFactory;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextFactory;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelKey;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentScopeHarnessInvoker;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotPayload;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.kernel.PersistedAgentKernelSnapshotResolver;
import com.stonewu.fusion.service.ai.run.kernel.RunConfigUnavailableException;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.Msg;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.List;
import java.util.Objects;
import java.util.Set;

@Component
public final class AgentExecutionFactory {

    private static final String SNAPSHOT_WHITELIST_VERSION = "snapshot-schema-2";

    private final AgentScopeHarnessInvoker harnessInvoker;
    private final AgentScopeRuntimeContextFactory runtimeContextFactory;
    private final AgentScopeEventMapper eventMapper;
    private final AiModelService modelService;
    private final AgentScopeModelFactory modelFactory;
    private final AgentRuntimeSchedulers schedulers;
    private final PersistedAgentKernelSnapshotResolver snapshotResolver;
    private final AiModelMetadataResolver modelMetadataResolver;

    public AgentExecutionFactory(
            AgentScopeHarnessInvoker harnessInvoker,
            AgentScopeRuntimeContextFactory runtimeContextFactory,
            AgentScopeEventMapper eventMapper,
            AiModelService modelService,
            AgentScopeModelFactory modelFactory,
            AgentRuntimeSchedulers schedulers,
            AiModelMetadataResolver modelMetadataResolver,
            ObjectMapper objectMapper) {
        this.harnessInvoker = Objects.requireNonNull(harnessInvoker, "harnessInvoker must not be null");
        this.runtimeContextFactory = Objects.requireNonNull(
                runtimeContextFactory, "runtimeContextFactory must not be null");
        this.eventMapper = Objects.requireNonNull(eventMapper, "eventMapper must not be null");
        this.modelService = Objects.requireNonNull(modelService, "modelService must not be null");
        this.modelFactory = Objects.requireNonNull(modelFactory, "modelFactory must not be null");
        this.schedulers = Objects.requireNonNull(schedulers, "schedulers must not be null");
        this.modelMetadataResolver = Objects.requireNonNull(
                modelMetadataResolver, "modelMetadataResolver must not be null");
        this.snapshotResolver = new PersistedAgentKernelSnapshotResolver(objectMapper);
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
                            .map(eventMapper::map)
                            .map(event -> childIdentity(event, run)),
                    ignored -> Mono.empty(),
                    () -> { });
        });
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
        if (!payload.tools().isEmpty()) {
            throw unavailable("Persisted tool implementations are not available for resume");
        }
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
        AgentKernelSnapshot validated = snapshotResolver.resolve(
                snapshot.snapshotJson(), snapshot.fingerprint(), modelVersion, List.of());
        requireSameModel(validated.payload(), model);
        AgentKernelKey key = AgentKernelKey.create(
                payload.agentDefinitionStableKey(),
                modelFingerprint,
                AgentKernelKey.promptVersion(
                        payload.systemPrompt(), payload.promptVariables()),
                List.of(),
                SNAPSHOT_WHITELIST_VERSION);
        return new AgentKernelSpec(
                key,
                model,
                payload.agentDefinitionStableKey(),
                payload.agentName(),
                payload.description(),
                payload.systemPrompt(),
                payload.promptVariables(),
                payload.maxIters(),
                List.of(),
                Set.of(),
                SNAPSHOT_WHITELIST_VERSION);
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
