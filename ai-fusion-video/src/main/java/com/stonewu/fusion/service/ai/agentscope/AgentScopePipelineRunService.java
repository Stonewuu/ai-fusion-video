package com.stonewu.fusion.service.ai.agentscope;

import cn.hutool.core.util.IdUtil;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.controller.ai.vo.AiReferenceVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.agentscope.context.ProjectContext;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentPromptVariables;
import com.stonewu.fusion.service.ai.agentscope.message.AgentScopeMessageMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.AgentExecutionRuntimeContextRequests;
import com.stonewu.fusion.service.ai.run.AgentRunCoordinator;
import com.stonewu.fusion.service.ai.run.AgentRunQueryService;
import com.stonewu.fusion.service.ai.run.AgentRunReplayService;
import com.stonewu.fusion.service.ai.run.AgentRuntimeInstanceIdentity;
import com.stonewu.fusion.service.ai.run.RunExecutionSupervisor;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshot;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import com.stonewu.fusion.service.ai.run.model.StartAgentRunCommand;
import com.stonewu.fusion.service.ai.run.model.StartedAgentRun;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** The only production entry point for starting a root AgentScope Harness run. */
@Service
public final class AgentScopePipelineRunService {

    private static final String DEFAULT_SYSTEM_PROMPT = """
            你是一个专业的 AI 视频创作助手，专注于帮助用户进行剧本编辑和分镜设计。

            你的能力包括：
            1. 剧本润色和改写
            2. 分镜脚本生成和编辑
            3. 角色对话优化
            4. 场景描述增强

            当用户提供引用内容时，请基于这些内容进行分析和处理。
            当需要修改内容时，请调用相应的工具来执行操作。

            请用中文回答，保持专业、简洁的风格。
            """;

    private final AiModelService modelService;
    private final AiAgentService agentService;
    private final AgentConversationService conversations;
    private final AgentKernelSpecFactory specFactory;
    private final AgentKernelSnapshotBuilder snapshots;
    private final AgentScopeMessageMapper messages;
    private final AgentRunCoordinator coordinator;
    private final AgentExecutionRuntimeContextRequests runtimeContexts;
    private final RunExecutionSupervisor supervisor;
    private final AgentRunQueryService runQueries;
    private final AgentRunReplayService replay;
    private final AgentRuntimeInstanceIdentity instanceIdentity;
    private final AgentScopeV2Properties properties;
    private final AgentRuntimeSchedulers schedulers;
    private final ObjectMapper objectMapper;

    public AgentScopePipelineRunService(
            AiModelService modelService,
            AiAgentService agentService,
            AgentConversationService conversations,
            AgentKernelSpecFactory specFactory,
            AgentKernelSnapshotBuilder snapshots,
            AgentScopeMessageMapper messages,
            AgentRunCoordinator coordinator,
            AgentExecutionRuntimeContextRequests runtimeContexts,
            RunExecutionSupervisor supervisor,
            AgentRunQueryService runQueries,
            AgentRunReplayService replay,
            AgentRuntimeInstanceIdentity instanceIdentity,
            AgentScopeV2Properties properties,
            AgentRuntimeSchedulers schedulers,
            ObjectMapper objectMapper) {
        this.modelService = Objects.requireNonNull(modelService, "modelService must not be null");
        this.agentService = Objects.requireNonNull(agentService, "agentService must not be null");
        this.conversations = Objects.requireNonNull(conversations, "conversations must not be null");
        this.specFactory = Objects.requireNonNull(specFactory, "specFactory must not be null");
        this.snapshots = Objects.requireNonNull(snapshots, "snapshots must not be null");
        this.messages = Objects.requireNonNull(messages, "messages must not be null");
        this.coordinator = Objects.requireNonNull(coordinator, "coordinator must not be null");
        this.runtimeContexts = Objects.requireNonNull(
                runtimeContexts, "runtimeContexts must not be null");
        this.supervisor = Objects.requireNonNull(supervisor, "supervisor must not be null");
        this.runQueries = Objects.requireNonNull(runQueries, "runQueries must not be null");
        this.replay = Objects.requireNonNull(replay, "replay must not be null");
        this.instanceIdentity = Objects.requireNonNull(
                instanceIdentity, "instanceIdentity must not be null");
        this.properties = Objects.requireNonNull(properties, "properties must not be null");
        this.schedulers = Objects.requireNonNull(schedulers, "schedulers must not be null");
        this.objectMapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
    }

    public Flux<AiChatStreamRespVO> stream(AiChatReqVO request, long userId) {
        return start(request, userId)
                .flatMap(run -> runQueries.requireAuthorizedRun(run.runId(), userId))
                .flatMapMany(run -> replay.replayThenLive(run.getRunId(), 0)
                        .concatMap(event -> runQueries.project(run, event)));
    }

    public Mono<StartedAgentRun> start(AiChatReqVO request, long userId) {
        if (userId <= 0) {
            return Mono.error(new IllegalArgumentException("userId must be positive"));
        }
        AiChatReqVO safeRequest = Objects.requireNonNull(request, "request must not be null");
        return Mono.fromCallable(() -> prepare(safeRequest, userId))
                .subscribeOn(schedulers.journal())
                .flatMap(prepared -> coordinator.start(prepared.admission())
                        .flatMap(started -> runtimeContexts.forRoot(
                                        started,
                                        prepared.spec().agentDefinitionStableKey(),
                                        prepared.project())
                                .flatMap(runtime -> supervisor.start(
                                                new StartAgentExecutionCommand(
                                                        started,
                                                        messages.toUserMessages(prepared.input()),
                                                        prepared.snapshot(),
                                                        prepared.spec(),
                                                        runtime))
                                        .thenReturn(started))));
    }

    private PreparedRun prepare(AiChatReqVO request, long userId) {
        String conversationId = normalize(request.getConversationId());
        if (conversationId == null) {
            conversationId = IdUtil.fastSimpleUUID();
            request.setConversationId(conversationId);
        }
        AiAgentDefinition definition = definition(request.getAgentType());
        Map<String, String> promptVariables = AgentPromptVariables.fromRequest(request);
        String visibleUserContent = userContent(request, definition, promptVariables);
        String input = input(request, visibleUserContent);
        String systemPrompt = systemPrompt(request, definition, promptVariables);
        AiModel model = model(request.getModelId());
        AgentKernelSpec spec = specFactory.createRoot(request, model, systemPrompt);
        AgentKernelSnapshot snapshot = snapshots.build(spec);
        Instant deadline = Instant.now()
                .plus(properties.getExecution().getRunTimeout())
                .truncatedTo(ChronoUnit.MILLIS);
        String title = title(request, definition, visibleUserContent);
        conversations.createOrUpdate(
                conversationId,
                userId,
                request.getProjectId(),
                request.getProjectId() == null ? null : "project",
                request.getProjectId(),
                request.getAgentType(),
                title,
                request.getCategory());

        String runId = IdUtil.fastSimpleUUID();
        StartAgentRunCommand admission = new StartAgentRunCommand(
                runId,
                conversationId,
                userId,
                request.getProjectId(),
                spec.agentDefinitionStableKey(),
                null,
                null,
                null,
                stateSessionId(conversationId, spec.agentDefinitionStableKey()),
                snapshot,
                instanceIdentity.value(),
                properties.getExecution().getOwnerLease(),
                deadline,
                visibleUserContent,
                normalize(request.getReferencesJson()));
        ProjectContext project = request.getProjectId() == null
                ? null
                : new ProjectContext(request.getProjectId());
        return new PreparedRun(spec, snapshot, admission, input, project);
    }

    private AiModel model(Long modelId) {
        AiModel model = modelId == null
                ? modelService.getDefaultByType(1)
                : modelService.getById(modelId);
        if (model == null) {
            throw new BusinessException(modelId == null
                    ? "未配置默认对话模型"
                    : "AI 模型不存在: " + modelId);
        }
        if (!Integer.valueOf(1).equals(model.getStatus())) {
            throw new BusinessException("AI 模型未启用: " + model.getId());
        }
        return model;
    }

    private AiAgentDefinition definition(String agentType) {
        String normalized = normalize(agentType);
        return normalized == null ? null : agentService.getRequiredByType(normalized);
    }

    private String systemPrompt(
            AiChatReqVO request,
            AiAgentDefinition definition,
            Map<String, String> promptVariables) {
        String prompt = normalize(request.getSystemPrompt());
        if (prompt == null) {
            prompt = definition == null
                    ? DEFAULT_SYSTEM_PROMPT.strip()
                    : requireText(definition.getSystemPrompt(), "agent.systemPrompt");
        }
        String instruction = normalize(request.getInstruction());
        if (instruction == null && definition != null) {
            instruction = normalize(definition.getInstructionTemplate());
        }
        if (instruction != null) {
            prompt = prompt + "\n\n" + AgentPromptVariables.render(
                    instruction, promptVariables);
        }
        return prompt;
    }

    private String userContent(
            AiChatReqVO request,
            AiAgentDefinition definition,
            Map<String, String> promptVariables) {
        String content = normalize(request.getMessage());
        if (content != null) {
            return request.getMessage();
        }
        if (definition == null || normalize(definition.getDefaultUserMessage()) == null) {
            throw new BusinessException("Agent 请求缺少用户消息");
        }
        return AgentPromptVariables.render(
                definition.getDefaultUserMessage(), promptVariables);
    }

    private String input(AiChatReqVO request, String userContent) {
        Map<String, Object> context = context(request);
        if (context.isEmpty()) {
            return userContent;
        }
        StringBuilder prompt = new StringBuilder("<references>\n");
        context.forEach((name, value) -> prompt
                .append("<reference name=\"").append(name).append("\">\n")
                .append(value).append("\n</reference>\n\n"));
        return prompt.append("</references>\n\n<user_request>\n")
                .append(userContent)
                .append("\n</user_request>")
                .toString();
    }

    private Map<String, Object> context(AiChatReqVO request) {
        Map<String, Object> context = new LinkedHashMap<>();
        String page = pageContext(request);
        if (page != null) {
            context.put("page_context", page);
        }
        if (request.getContext() != null) {
            context.putAll(request.getContext());
        }
        if (request.getReferences() != null) {
            for (AiReferenceVO reference : request.getReferences()) {
                context.put(referenceKey(reference), referenceContent(reference));
            }
        }
        return context;
    }

    private String pageContext(AiChatReqVO request) {
        if ((request.getAutoReferences() == null || request.getAutoReferences().isEmpty())
                && request.getProjectId() == null) {
            return null;
        }
        StringBuilder value = new StringBuilder("<page_context>\n");
        if (request.getProjectId() != null) {
            value.append("  <project_id>").append(request.getProjectId())
                    .append("</project_id>\n");
        }
        if (request.getAutoReferences() != null) {
            for (AiReferenceVO reference : request.getAutoReferences()) {
                String type = requireText(reference.getType(), "autoReference.type");
                Long id = Objects.requireNonNull(reference.getId(), "autoReference.id must not be null");
                value.append("  <").append(type).append("_id>")
                        .append(id).append("</").append(type).append("_id>\n");
            }
        }
        return value.append("</page_context>").toString();
    }

    private String referenceKey(AiReferenceVO reference) {
        Objects.requireNonNull(reference, "reference must not be null");
        String title = normalize(reference.getTitle());
        return title != null
                ? title
                : requireText(reference.getType(), "reference.type") + '_'
                        + Objects.requireNonNull(reference.getId(), "reference.id must not be null");
    }

    private String referenceContent(AiReferenceVO reference) {
        String metadata = normalize(reference.getMetadata());
        String type = requireText(reference.getType(), "reference.type");
        Long id = Objects.requireNonNull(reference.getId(), "reference.id must not be null");
        if (metadata == null) {
            return "<meta>\n  <type>" + type + "</type>\n  <id>" + id
                    + "</id>\n  <hint>请使用查询工具获取完整内容</hint>\n</meta>";
        }
        try {
            JsonNode node = objectMapper.readTree(metadata);
            if (node == null || !node.isObject()) {
                throw new IllegalArgumentException("reference.metadata must be a JSON object");
            }
            JsonNode fullText = node.get("fullText");
            if (fullText == null || !fullText.isTextual() || fullText.textValue().isBlank()) {
                return "<meta>\n  <type>" + type + "</type>\n  <id>" + id
                        + "</id>\n  <hint>请使用查询工具获取完整内容</hint>\n</meta>";
            }
            return "<meta>\n  <type>" + type + "</type>\n  <id>" + id
                    + "</id>\n</meta>\n<content>\n" + fullText.textValue()
                    + "\n</content>";
        } catch (JsonProcessingException invalidMetadata) {
            throw new IllegalArgumentException("reference.metadata is invalid JSON", invalidMetadata);
        }
    }

    private String title(
            AiChatReqVO request,
            AiAgentDefinition definition,
            String userContent) {
        String title = normalize(request.getTitle());
        if (title == null && normalize(request.getMessage()) != null) {
            title = request.getMessage();
        }
        if (title == null && definition != null) {
            title = definition.getName();
        }
        if (title == null) {
            title = userContent;
        }
        // Titles are generated once from the first visible user prompt. Keep
        // whitespace deterministic so subsequent requests cannot produce a
        // different title merely because of formatting in the prompt.
        title = title.replaceAll("\\s+", " ").trim();
        return title.length() <= 50 ? title : title.substring(0, 50);
    }

    private String stateSessionId(String conversationId, String definitionKey) {
        return "afv:v2:" + conversationId + ':' + definitionKey;
    }

    private static String normalize(String value) {
        return value == null || value.isBlank() ? null : value.trim();
    }

    private static String requireText(String value, String field) {
        String normalized = normalize(value);
        if (normalized == null) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        return normalized;
    }

    private record PreparedRun(
            AgentKernelSpec spec,
            AgentKernelSnapshot snapshot,
            StartAgentRunCommand admission,
            String input,
            ProjectContext project) {
        private PreparedRun {
            Objects.requireNonNull(spec, "spec must not be null");
            Objects.requireNonNull(snapshot, "snapshot must not be null");
            Objects.requireNonNull(admission, "admission must not be null");
            requireText(input, "input");
        }
    }
}
