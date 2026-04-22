package com.stonewu.fusion.service.ai.agentscope;

import cn.hutool.core.collection.CollUtil;
import cn.hutool.core.util.IdUtil;
import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.controller.ai.vo.AiReferenceVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AgentMessageService;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.AiStreamRedisService;
import com.stonewu.fusion.service.ai.AiToolConfigService;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import io.agentscope.core.ReActAgent;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.model.ExecutionConfig;
import io.agentscope.core.model.Model;
import io.agentscope.core.session.mysql.MysqlSession;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.core.tool.ToolkitConfig;
import io.agentscope.core.tool.subagent.SubAgentConfig;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;
import reactor.core.Disposable;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.SignalType;
import reactor.core.scheduler.Schedulers;
import com.stonewu.fusion.service.ai.AiStreamRedisService.StreamEventAccumulator;
import com.stonewu.fusion.service.ai.AiStreamRedisService.StreamEventAccumulator.AccumulatedEvent;
import reactor.core.publisher.Sinks;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * AgentScope 版 AI 助手服务
 * <p>
 * 基于 AgentScope Java ReActAgent 实现流式对话，支持：
 * - 父/子 Agent（Multi-Agent）架构
 * - 通过 Hook 系统实现流式事件推送
 * - 子 Agent 事件通过 forwardEvents 穿透 + parentToolCallId 映射
 * - 工具和子 Agent 的并行调用（Toolkit.parallel=true）
 * - Redis Stream 解耦，支持 SSE 断线重连
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AgentScopeAssistantService {

    private final AiModelService aiModelService;
    private final AiAgentService aiAgentService;
    private final AiToolConfigService aiToolConfigService;
    private final AgentConversationService conversationService;
    private final AgentMessageService messageService;
    private final AgentScopeModelFactory agentScopeModelFactory;
    private final StringRedisTemplate stringRedisTemplate;
    private final AiStreamRedisService aiStreamRedisService;
    private final javax.sql.DataSource dataSource;

    /** AgentScope MySQL Session（子 Agent 会话持久化） */
    private MysqlSession mysqlSession;

    @PostConstruct
    public void init() {
        this.mysqlSession = new MysqlSession(dataSource, true);
        log.info("AgentScope MysqlSession 初始化完成（createIfNotExist=true）");
    }

    /** 保存后台 Redis 事件流的 Disposable */
    private final ConcurrentHashMap<String, Disposable> activeSubscriptions = new ConcurrentHashMap<>();
    /** 保存 agent.call() 的 Disposable，cancelStream 时真正取消 Agent 执行 */
    private final ConcurrentHashMap<String, Disposable> agentCallSubscriptions = new ConcurrentHashMap<>();
    /** 保存当前对话的 StreamingEventHook，供 cancelStream 时中断主/子 Agent */
    private final ConcurrentHashMap<String, StreamingEventHook> activeStreamingHooks = new ConcurrentHashMap<>();

    private static final String CANCEL_FLAG_KEY = "fv:agent:cancel:";
    private static final Duration CANCEL_FLAG_TTL = Duration.ofHours(1);

    private static final String DEFAULT_SYSTEM_PROMPT = """
            你是一个专业的AI视频创作助手，专注于帮助用户进行剧本编辑和分镜设计。

            你的能力包括：
            1. 剧本润色和改写
            2. 分镜脚本生成和编辑
            3. 角色对话优化
            4. 场景描述增强

            当用户提供引用内容时，请基于这些内容进行分析和处理。
            当需要修改内容时，请调用相应的工具来执行操作。

            请用中文回答，保持专业、简洁的风格。
            """;

    /**
     * 流式对话（AgentScope 版）
     */
    public Flux<AiChatStreamRespVO> stream(AiChatReqVO reqVO, Long userId) {
        log.info("[AgentScope:stream] 开始流式调用: message={}, conversationId={}, agentType={}",
                reqVO.getMessage(), reqVO.getConversationId(), reqVO.getAgentType());

        String conversationId = StrUtil.blankToDefault(reqVO.getConversationId(), IdUtil.fastSimpleUUID());
        String messageId = IdUtil.fastSimpleUUID();
        String mainAgentName = "ai_assistant_agent";

        // 新一轮执行开始前清理上一次取消留下的 Redis 标志，避免误伤新的同会话请求。
        clearCancelFlag(conversationId);

        try {
            // 1. 获取 AgentScope Model
            Model model = getAgentScopeModel(reqVO.getModelId());

            // 2. 获取系统提示词
            String systemPrompt = getSystemPrompt(reqVO);

            // 3. 构建工具执行上下文
            ToolExecutionContext toolExecContext = ToolExecutionContext.builder()
                    .userId(userId)
                    .ownerType(1)
                    .ownerId(userId)
                    .build();

            // 4. 创建事件 Sink（用于 Hook 推送事件）
            Sinks.Many<AiChatStreamRespVO> eventSink = Sinks.many().multicast()
                    .onBackpressureBuffer(1024);

            // 5. 创建取消令牌（基于 Redis 标志，集群安全）
            AgentCancellationToken cancellationToken = new AgentCancellationToken(
                    () -> isCancelled(conversationId));

            // 6. 创建 StreamingEventHook
            StreamingEventHook streamingHook = new StreamingEventHook(
                    eventSink, conversationId, messageId, mainAgentName, cancellationToken);
                activeStreamingHooks.put(conversationId, streamingHook);

            // 7. 构建 Toolkit（普通工具 + 子 Agent 工具）
            Toolkit toolkit = buildToolkit(reqVO, model, toolExecContext, streamingHook, cancellationToken);

            // 7. 构建 ReActAgent
            ReActAgent.Builder agentBuilder = ReActAgent.builder()
                    .name(mainAgentName)
                    .sysPrompt(systemPrompt)
                    .model(model)
                    .maxIters(999)
                    .hooks(List.of(streamingHook));

            if (toolkit != null) {
                agentBuilder.toolkit(toolkit);
            }

            ReActAgent agent = agentBuilder.build();
            streamingHook.registerActiveAgent(agent);

            // 8. 构建输入消息（需要在记录对话前构建，用于标题回退和保存）
            String inputMessage = buildInputMessage(reqVO);

            // 9. 记录对话索引
            String title = StrUtil.isNotBlank(reqVO.getTitle())
                    ? reqVO.getTitle()
                    : StrUtil.isNotBlank(reqVO.getMessage())
                            ? StrUtil.sub(reqVO.getMessage(), 0, 50)
                            : null;
            // 前端既没有传 title 也没传 message 时（Pipeline 场景），使用 Agent 名称
            if (StrUtil.isBlank(title) && StrUtil.isNotBlank(reqVO.getAgentType())) {
                AiAgentDefinition agentConfig = aiAgentService.getByType(reqVO.getAgentType());
                if (agentConfig != null && StrUtil.isNotBlank(agentConfig.getName())) {
                    title = agentConfig.getName();
                }
            }
            if (StrUtil.isBlank(title)) {
                title = StrUtil.sub(inputMessage, 0, 50);
            }
            conversationService.createOrUpdate(
                    conversationId, userId, reqVO.getProjectId(),
                    null, null, reqVO.getAgentType(), title, reqVO.getCategory());

            // 10. 保存用户消息（使用构建后的完整输入内容）
            messageService.saveUserMessage(conversationId,
                    StrUtil.isNotBlank(reqVO.getMessage()) ? reqVO.getMessage() : inputMessage,
                    reqVO.getReferencesJson());

            // 11. 构建 Agent 输入
            Msg userMsg = Msg.builder()
                    .role(MsgRole.USER)
                    .textContent(inputMessage)
                    .build();

            // 11. Redis Stream 初始化
            aiStreamRedisService.cleanup(conversationId);
            aiStreamRedisService.markActive(conversationId);
            AtomicReference<String> terminalStatus = new AtomicReference<>("running");

            // 12. 用于累积主 Agent 的 REASONING 和 CONTENT 增量文本
            StringBuilder reasoningAccumulator = new StringBuilder();
            StringBuilder contentAccumulator = new StringBuilder();
            // 记录思考耗时
            long[] reasoningDuration = { 0L };

                // 13. 创建事件合并累积器（用于 Replay List）
            StreamEventAccumulator accumulator = new StreamEventAccumulator(conversationId);

                // 14. 先挂上事件订阅，再启动 Agent，避免早期 reasoning token 在无订阅者时丢失。
                // 同时将 Redis / DB 写入切到 boundedElastic，避免阻塞 Hook 线程导致上游 chunk 堆积。
            Disposable redisSubscription = eventSink.asFlux()
                    .publishOn(Schedulers.boundedElastic())
                    .takeWhile(event -> !isCancelled(conversationId))
                    .doOnNext(event -> {
                        String outputType = event.getOutputType();
                        if ("ERROR".equals(outputType)) {
                            terminalStatus.set("failed");
                        } else if ("CANCELLED".equals(outputType)) {
                            terminalStatus.set("cancelled");
                        } else if ("DONE".equals(outputType)) {
                            terminalStatus.compareAndSet("running", "completed");
                        }

                        // 写入 Redis Stream（实时通道，逐 token）
                        String streamId = aiStreamRedisService.publish(conversationId, event);

                        // 通过 Accumulator 合并后写入 Replay List（重连通道）
                        if (streamId != null) {
                            List<AccumulatedEvent> flushed = accumulator.accumulate(event, streamId);
                            for (AccumulatedEvent ae : flushed) {
                                aiStreamRedisService.appendReplayEvent(
                                        conversationId, ae.getEvent(), ae.getLastStreamId());
                            }
                        }

                        // 持久化中间步骤到数据库
                        persistStreamEvent(conversationId, event,
                                reasoningAccumulator, contentAccumulator, reasoningDuration);
                    })
                    .doOnComplete(() -> {
                        log.info("[AgentScope:stream] 事件流完成: {}", conversationId);
                        // flush 累积器残余
                        AccumulatedEvent remaining = accumulator.flush();
                        if (remaining != null) {
                            aiStreamRedisService.appendReplayEvent(
                                    conversationId, remaining.getEvent(), remaining.getLastStreamId());
                        }
                        if ("failed".equals(terminalStatus.get())) {
                            aiStreamRedisService.markError(conversationId);
                        } else {
                            aiStreamRedisService.markCompleted(conversationId);
                        }
                        aiStreamRedisService.scheduleCleanup(conversationId);
                    })
                    .doOnError(e -> {
                        log.error("[AgentScope:stream] 事件流出错", e);
                        // flush 累积器残余
                        AccumulatedEvent remaining = accumulator.flush();
                        if (remaining != null) {
                            aiStreamRedisService.appendReplayEvent(
                                    conversationId, remaining.getEvent(), remaining.getLastStreamId());
                        }
                        AiChatStreamRespVO errorEvent = new AiChatStreamRespVO()
                                .setConversationId(conversationId)
                                .setOutputType("ERROR")
                                .setError(e.getMessage())
                                .setFinished(true);
                        String errorStreamId = aiStreamRedisService.publish(conversationId, errorEvent);
                        aiStreamRedisService.appendReplayEvent(conversationId, errorEvent,
                                errorStreamId != null ? errorStreamId : "0-0");
                        aiStreamRedisService.markError(conversationId);
                        aiStreamRedisService.scheduleCleanup(conversationId);
                    })
                    .doFinally(signalType -> {
                        activeSubscriptions.remove(conversationId);
                    })
                    .subscribe();

            activeSubscriptions.put(conversationId, redisSubscription);

                    // 15. 后台异步调用 Agent
                    // agent.call() 返回 Mono<Msg>，Hook 会在执行过程中推送事件
                    Disposable agentDisposable = agent.call(userMsg)
                        .doOnSuccess(response -> {
                        terminalStatus.compareAndSet("running", "completed");
                        // 发送 DONE 事件（不再在此保存 assistant 消息，改为在事件流中根据累积文本保存）
                        eventSink.tryEmitNext(new AiChatStreamRespVO()
                            .setMessageId(messageId)
                            .setConversationId(conversationId)
                            .setOutputType("DONE")
                            .setFinished(true));
                        eventSink.tryEmitComplete();
                        })
                        .doOnError(e -> {
                        terminalStatus.set("failed");
                        log.error("[AgentScope:stream] Agent 调用出错", e);
                        eventSink.tryEmitNext(new AiChatStreamRespVO()
                            .setMessageId(messageId)
                            .setConversationId(conversationId)
                            .setOutputType("ERROR")
                            .setError(e.getMessage())
                            .setFinished(true));
                        eventSink.tryEmitComplete();
                        })
                        .doFinally(signalType -> {
                        agentCallSubscriptions.remove(conversationId);
                        activeStreamingHooks.remove(conversationId);
                        streamingHook.clearTrackedAgents();

                        boolean wasCancelled = signalType == SignalType.CANCEL
                                || "cancelled".equals(terminalStatus.get())
                                || isCancelled(conversationId);

                        String finalStatus;
                        if (signalType == SignalType.ON_ERROR || "failed".equals(terminalStatus.get())) {
                            finalStatus = wasCancelled ? "cancelled" : "failed";
                        } else if (wasCancelled) {
                            finalStatus = "cancelled";
                        } else {
                            finalStatus = "completed";
                        }
                        conversationService.finish(conversationId, finalStatus);
                    })
                        .onErrorResume(e -> Mono.empty())
                        .subscribe();
                    agentCallSubscriptions.put(conversationId, agentDisposable);

            // SSE 从 Redis Stream 读取（实时逐 token）
            return aiStreamRedisService.subscribe(conversationId);

        } catch (Throwable e) {
            activeStreamingHooks.remove(conversationId);
            log.error("[AgentScope:stream] 初始化失败", e);
            try {
                conversationService.finish(conversationId, "failed");
                aiStreamRedisService.markError(conversationId);
                aiStreamRedisService.scheduleCleanup(conversationId);
            } catch (Exception statusError) {
                log.warn("[AgentScope:stream] 初始化失败后的状态回写失败: {}", conversationId, statusError);
            }
            return Flux.just(new AiChatStreamRespVO()
                    .setMessageId(IdUtil.fastSimpleUUID())
                    .setConversationId(conversationId)
                    .setOutputType("ERROR")
                    .setFinished(true)
                    .setError("AgentScope Agent 调用失败: " + e.getMessage()));
        }
    }

    /**
     * 取消对话
     */
    public void cancelStream(String conversationId) {
        if (StrUtil.isBlank(conversationId)) {
            return;
        }
        String cancelKey = CANCEL_FLAG_KEY + conversationId;
        stringRedisTemplate.opsForValue().set(cancelKey, "1", CANCEL_FLAG_TTL);
        log.info("[cancelStream] 已设置取消标志: {}", conversationId);

        StreamingEventHook streamingHook = activeStreamingHooks.get(conversationId);
        if (streamingHook != null) {
            streamingHook.interruptTrackedAgents();
        }

        // 1. 取消 agent.call() 的底层执行（LLM 调用 + 工具调用）
        Disposable agentSub = agentCallSubscriptions.remove(conversationId);
        if (agentSub != null && !agentSub.isDisposed()) {
            agentSub.dispose();
            log.info("已取消后台 AgentScope Agent 调用: conversationId={}", conversationId);
        }

        // 2. 取消 Redis 事件流订阅
        Disposable sub = activeSubscriptions.remove(conversationId);
        if (sub != null && !sub.isDisposed()) {
            sub.dispose();
            log.info("已取消后台 AgentScope 流订阅: conversationId={}", conversationId);
        }

        conversationService.finish(conversationId, "cancelled");

        AiChatStreamRespVO cancelledEvent = new AiChatStreamRespVO()
                .setConversationId(conversationId)
                .setOutputType("CANCELLED")
                .setContent("对话已停止")
                .setFinished(true);
        String cancelStreamId = aiStreamRedisService.publish(conversationId, cancelledEvent);
        aiStreamRedisService.appendReplayEvent(conversationId, cancelledEvent,
                cancelStreamId != null ? cancelStreamId : "0-0");
        aiStreamRedisService.markCompleted(conversationId);
        aiStreamRedisService.scheduleCleanup(conversationId);
    }

    /**
     * 重连已有流式对话
     * <p>
     * 双通道回放：先从 Replay List 发送合并后的完整历史事件（瞬间回放），
     * 然后从 Redis Stream 的 lastStreamId 位置续传实时 token 事件（逐字流式）。
     */
    public Flux<AiChatStreamRespVO> reconnectStream(String conversationId) {
        log.info("[reconnectStream] 重连: conversationId={}", conversationId);
        String status = aiStreamRedisService.getStatus(conversationId);
        if ("NONE".equals(status) || "COMPLETED".equals(status) || "ERROR".equals(status)) {
            return Flux.empty();
        }

        // 1. 从 Replay List 读取合并后的历史事件
        AiStreamRedisService.ReplayResult replayResult =
                aiStreamRedisService.getReplayEvents(conversationId);
        Flux<AiChatStreamRespVO> historyFlux = Flux.fromIterable(replayResult.getEvents());

        // 2. 从 Redis Stream 的 lastStreamId 位置续传实时 token
        String lastStreamId = replayResult.getLastStreamId();
        log.info("[reconnectStream] 回放 {} 条合并历史，从 Stream {} 续传",
                replayResult.getEvents().size(), lastStreamId);
        Flux<AiChatStreamRespVO> liveFlux =
                aiStreamRedisService.subscribeFrom(conversationId, lastStreamId);

        // 3. 先回放历史，再续传实时
        return Flux.concat(historyFlux, liveFlux);
    }

    public String getStreamStatus(String conversationId) {
        return aiStreamRedisService.getStatus(conversationId);
    }

    // ========== 事件持久化 ==========

    /**
     * 将 SSE 流中的中间步骤持久化到数据库，确保历史记录面板能查看完整的推理和工具调用过程。
     * <p>
     * 持久化策略：
     * - TOOL_CALL：保存工具调用开始记录（主 Agent 和子 Agent 的工具调用都保存）
     * - TOOL_FINISHED：保存工具调用结果
     * - REASONING：累积思考文本（在 DONE 时随 assistant 消息一起保存）
     * - CONTENT：累积回复文本（在 DONE 时保存为 assistant 消息）
     * - DONE：保存累积的 assistant 消息（含 reasoning）
     */
    private void persistStreamEvent(String conversationId, AiChatStreamRespVO event,
            StringBuilder reasoningAccumulator,
            StringBuilder contentAccumulator,
            long[] reasoningDuration) {
        try {
            String outputType = event.getOutputType();
            if (outputType == null) {
                return;
            }

            // 是否为子 Agent 事件
            boolean isSubAgent = StrUtil.isNotBlank(event.getParentToolCallId());

            switch (outputType) {
                case "REASONING" -> {
                    // 仅累积主 Agent 的思考文本（子 Agent 的 reasoning 不在主对话中保存）
                    if (!isSubAgent && StrUtil.isNotBlank(event.getReasoningContent())) {
                        reasoningAccumulator.append(event.getReasoningContent());
                    }
                }

                case "CONTENT" -> {
                    // 记录思考耗时
                    if (!isSubAgent && event.getReasoningDurationMs() != null) {
                        reasoningDuration[0] = event.getReasoningDurationMs();
                    }
                    // 累积主 Agent 的内容文本（保留空白字符，仅跳过 null/空串）
                    if (!isSubAgent && StrUtil.isNotEmpty(event.getContent())) {
                        contentAccumulator.append(event.getContent());
                    }
                }

                case "TOOL_CALL" -> {
                    // 工具调用前，先将已累积的模型输出保存为 assistant 消息
                    // 这样 ReAct 循环中每次工具调用前的中间思考和输出都会被记录
                    if (!isSubAgent) {
                        flushAssistantMessage(conversationId, reasoningAccumulator,
                                contentAccumulator, reasoningDuration);
                    }
                    // 保存每个工具调用的发起记录（status=running）
                    if (event.getToolCalls() != null) {
                        for (AiChatStreamRespVO.ToolCallVO tc : event.getToolCalls()) {
                            messageService.saveToolCall(conversationId, tc.getName(),
                                    "running", tc.getArguments(),
                                    tc.getId(), event.getParentToolCallId());
                        }
                    }
                }

                case "TOOL_FINISHED" -> {
                    // 保存工具调用结果（status=success/error）
                    if (StrUtil.isNotBlank(event.getToolName())) {
                        String status = "error".equals(event.getToolStatus()) ? "error" : "success";
                        messageService.saveToolCall(conversationId, event.getToolName(),
                                status, event.getToolResult(),
                                event.getToolCallId(), event.getParentToolCallId());
                    }
                }

                case "DONE" -> {
                    // 保存最后一段累积的 assistant 消息（含 reasoning）
                    flushAssistantMessage(conversationId, reasoningAccumulator,
                            contentAccumulator, reasoningDuration);
                }

                default -> {
                    // ERROR, CANCELLED 等不需要额外持久化
                }
            }
        } catch (Exception e) {
            // 持久化失败不应影响主流程
            log.warn("[persistStreamEvent] 保存中间步骤失败 conversationId={}, type={}",
                    conversationId, event.getOutputType(), e);
        }
    }

    /**
     * 将累积的 REASONING + CONTENT 刷新为一条 assistant 消息并清空累积器。
     * 如果累积内容为空则跳过，不生成空消息。
     */
    private void flushAssistantMessage(String conversationId,
            StringBuilder reasoningAccumulator,
            StringBuilder contentAccumulator,
            long[] reasoningDuration) {
        String content = contentAccumulator.toString();
        String reasoning = reasoningAccumulator.length() > 0
                ? reasoningAccumulator.toString()
                : null;
        Long duration = reasoningDuration[0] > 0 ? reasoningDuration[0] : null;

        if (StrUtil.isNotEmpty(content) || reasoning != null) {
            messageService.saveAssistantMessage(conversationId, content, reasoning, duration);
        }

        // 清空累积器，为下一段输出做准备
        contentAccumulator.setLength(0);
        reasoningAccumulator.setLength(0);
        reasoningDuration[0] = 0L;
    }

    // ========== 私有方法 ==========

    private Model getAgentScopeModel(Long modelId) {
        AiModel aiModel;
        if (modelId != null) {
            aiModel = aiModelService.getById(modelId);
            if (aiModel == null) {
                throw new BusinessException("AI 模型不存在: " + modelId);
            }
        } else {
            aiModel = aiModelService.getDefaultByType(1);
            if (aiModel == null) {
                throw new BusinessException("未配置默认对话模型");
            }
        }
        return agentScopeModelFactory.getOrCreate(aiModel);
    }

    private String getSystemPrompt(AiChatReqVO reqVO) {
        String systemPrompt;
        if (StrUtil.isNotBlank(reqVO.getSystemPrompt())) {
            systemPrompt = reqVO.getSystemPrompt();
        } else if (StrUtil.isNotBlank(reqVO.getAgentType())) {
            AiAgentDefinition agentConfig = aiAgentService.getByType(reqVO.getAgentType());
            if (agentConfig != null && StrUtil.isNotBlank(agentConfig.getSystemPrompt())) {
                systemPrompt = agentConfig.getSystemPrompt();
            } else {
                systemPrompt = DEFAULT_SYSTEM_PROMPT;
            }
        } else {
            systemPrompt = DEFAULT_SYSTEM_PROMPT;
        }

        // 拼接 instructionTemplate（含 {projectId} 等模板变量替换）
        if (StrUtil.isNotBlank(reqVO.getAgentType())) {
            AiAgentDefinition agentConfig = aiAgentService.getByType(reqVO.getAgentType());
            if (agentConfig != null && StrUtil.isNotBlank(agentConfig.getInstructionTemplate())) {
                String instruction = replaceTemplateVariables(agentConfig.getInstructionTemplate(), reqVO);
                systemPrompt = systemPrompt + "\n\n" + instruction;
            }
        }

        return systemPrompt;
    }

    /**
     * 构建 Toolkit（含普通工具和子 Agent 工具）
     */
    private Toolkit buildToolkit(AiChatReqVO reqVO, Model model,
            ToolExecutionContext toolExecContext,
            StreamingEventHook streamingHook,
            AgentCancellationToken cancellationToken) {
        String agentType = reqVO.getAgentType();

        // 检查是否启用工具
        if (StrUtil.isNotBlank(agentType)) {
            AiAgentDefinition agentDef = aiAgentService.getByType(agentType);
            if (agentDef != null && !Integer.valueOf(1).equals(agentDef.getEnableTools())) {
                return null;
            }
        }

        // 获取工具列表
        List<ToolExecutor> enabledTools = aiToolConfigService.getEnabledToolsByAgent(agentType);
        List<AiAgentDefinition.SubAgentToolDef> subAgentTools = aiToolConfigService.getSubAgentTools(agentType);

        // 与前端 enabledTools 取交集
        List<ToolExecutor> filteredTools = new ArrayList<>();
        for (ToolExecutor tool : enabledTools) {
            if (CollUtil.isNotEmpty(reqVO.getEnabledTools())
                    && !reqVO.getEnabledTools().contains(tool.getToolName())) {
                continue;
            }
            filteredTools.add(tool);
        }

        List<AiAgentDefinition.SubAgentToolDef> filteredSubAgents = new ArrayList<>();
        for (AiAgentDefinition.SubAgentToolDef subTool : subAgentTools) {
            if (CollUtil.isNotEmpty(reqVO.getEnabledTools())
                    && !reqVO.getEnabledTools().contains(subTool.getToolName())) {
                continue;
            }
            filteredSubAgents.add(subTool);
        }

        if (filteredTools.isEmpty() && filteredSubAgents.isEmpty()) {
            return null;
        }

        // 创建 Toolkit（启用并行执行）
        Toolkit toolkit = new Toolkit(ToolkitConfig.builder()
                .parallel(true) // 并行执行多工具
                .executionConfig(ExecutionConfig.builder()
                        .timeout(Duration.ofMinutes(20))
                        .build())
                .build());

        // 注册普通工具（通过 AgentScopeToolAdapter 适配）
        for (ToolExecutor tool : filteredTools) {
            AgentScopeToolAdapter adapter = new AgentScopeToolAdapter(tool, toolExecContext, cancellationToken);
            toolkit.registerAgentTool(adapter);
        }

        // 注册子 Agent 工具
        for (AiAgentDefinition.SubAgentToolDef subAgentToolDef : filteredSubAgents) {
            registerSubAgentTool(toolkit, subAgentToolDef, model, reqVO, toolExecContext, streamingHook, cancellationToken);
        }

        log.info("AgentScope Toolkit 构建完成: 普通工具={}, 子Agent工具={}, 总计={}",
                filteredTools.size(), filteredSubAgents.size(),
                filteredTools.size() + filteredSubAgents.size());

        return toolkit;
    }

    /**
     * 注册子 Agent 工具
     */
    private void registerSubAgentTool(Toolkit toolkit,
            AiAgentDefinition.SubAgentToolDef subAgentToolDef,
            Model model,
            AiChatReqVO reqVO,
            ToolExecutionContext toolExecContext,
            StreamingEventHook streamingHook,
            AgentCancellationToken cancellationToken) {
        try {
            String subAgentType = subAgentToolDef.getRefAgentType();
            AiAgentDefinition subAgentDef = aiAgentService.getByType(subAgentType);

            // 确定 systemPrompt
            String systemPrompt = StrUtil.isNotBlank(subAgentToolDef.getSystemPromptOverride())
                    ? subAgentToolDef.getSystemPromptOverride()
                    : (subAgentDef != null ? subAgentDef.getSystemPrompt() : "你是一个AI助手。");

            // 确定 instruction
            String instruction = StrUtil.isNotBlank(subAgentToolDef.getInstructionOverride())
                    ? subAgentToolDef.getInstructionOverride()
                    : (subAgentDef != null ? subAgentDef.getInstructionTemplate() : null);

            if (StrUtil.isNotBlank(instruction)) {
                instruction = replaceTemplateVariables(instruction, reqVO);
                systemPrompt = systemPrompt + "\n\n" + instruction;
            }

            // 子 Agent 的内部工具
            Toolkit subToolkit = null;
            if (subAgentDef != null && Integer.valueOf(1).equals(subAgentDef.getEnableTools())) {
                List<ToolExecutor> subTools = aiToolConfigService.getEnabledToolsByAgent(subAgentType);
                if (!subTools.isEmpty()) {
                    subToolkit = new Toolkit(ToolkitConfig.builder()
                            .parallel(true)
                            .executionConfig(ExecutionConfig.builder()
                                    .timeout(Duration.ofMinutes(20))
                                    .build())
                            .build());
                    for (ToolExecutor subTool : subTools) {
                        subToolkit.registerAgentTool(new AgentScopeToolAdapter(subTool, toolExecContext, cancellationToken));
                    }
                }
            }

            final String finalSysPrompt = systemPrompt;
            final Toolkit finalSubToolkit = subToolkit;

            String toolName = subAgentToolDef.getToolName();

            SubAgentConfig config = SubAgentConfig.builder()
                    .toolName(toolName)
                    .description(subAgentToolDef.getDescription())
                    .forwardEvents(true) // 关键：转发子 Agent 事件到父 Hook 链
                    .session(mysqlSession) // 使用 MySQL 持久化子 Agent 会话
                    .build();

            // 注册子 Agent 工具名到 Hook，以便自动建立 parentToolCallId 映射
            streamingHook.registerSubAgentToolName(toolName);

            toolkit.registration()
                    .subAgent(() -> {
                        ReActAgent.Builder subBuilder = ReActAgent.builder()
                                .name(subAgentToolDef.getToolName())
                                .sysPrompt(finalSysPrompt)
                                .model(model) // 复用父 Agent 模型
                                .maxIters(999)
                                .hooks(List.of(streamingHook)); // 共享 Hook 实例

                        if (finalSubToolkit != null) {
                            subBuilder.toolkit(finalSubToolkit);
                        }

                        ReActAgent subAgent = subBuilder.build();
                        streamingHook.registerActiveAgent(subAgent);
                        return subAgent;
                    }, config)
                    .apply();

            log.info("子 Agent 注册完成: name={}, toolName={}, hasSubTools={}",
                    subAgentToolDef.getToolName(), config.getToolName(),
                    subToolkit != null);

        } catch (Exception e) {
            log.error("注册子 Agent 工具失败: name={}", subAgentToolDef.getToolName(), e);
        }
    }

    /**
     * 替换模板中的命名占位符
     */
    private String replaceTemplateVariables(String template, AiChatReqVO reqVO) {
        if (StrUtil.isBlank(template)) {
            return template;
        }

        if (reqVO.getProjectId() != null) {
            template = template.replace("{projectId}", String.valueOf(reqVO.getProjectId()));
        }

        if (CollUtil.isNotEmpty(reqVO.getAutoReferences())) {
            for (AiReferenceVO ref : reqVO.getAutoReferences()) {
                if (ref.getId() != null && StrUtil.isNotBlank(ref.getType())) {
                    template = template.replace("{" + ref.getType() + "Id}",
                            String.valueOf(ref.getId()));
                }
            }
        }

        if (reqVO.getContext() != null && template.contains("{")) {
            for (Map.Entry<String, Object> entry : reqVO.getContext().entrySet()) {
                String placeholder = "{" + entry.getKey() + "}";
                if (entry.getValue() != null && template.contains(placeholder)) {
                    template = template.replace(placeholder, String.valueOf(entry.getValue()));
                }
            }
        }

        return template;
    }

    /**
     * 构建输入消息（含上下文引用）
     * <p>
     * 当前端未传 message 时，自动从 Agent 定义的 defaultUserMessage 模板生成。
     */
    private String buildInputMessage(AiChatReqVO reqVO) {
        StringBuilder message = new StringBuilder();

        Map<String, Object> fullContext = buildContext(reqVO);

        if (!fullContext.isEmpty()) {
            message.append("<references>\n");
            for (Map.Entry<String, Object> entry : fullContext.entrySet()) {
                message.append("<reference name=\"").append(entry.getKey()).append("\">\n");
                message.append(entry.getValue());
                message.append("\n</reference>\n\n");
            }
            message.append("</references>\n\n");
            message.append("<user_request>\n");
        }

        // 获取用户消息：优先使用前端传入的 message，否则使用 Agent 的 defaultUserMessage 模板
        String userMessage = reqVO.getMessage();
        if (StrUtil.isBlank(userMessage) && StrUtil.isNotBlank(reqVO.getAgentType())) {
            AiAgentDefinition agentConfig = aiAgentService.getByType(reqVO.getAgentType());
            if (agentConfig != null && StrUtil.isNotBlank(agentConfig.getDefaultUserMessage())) {
                userMessage = replaceTemplateVariables(agentConfig.getDefaultUserMessage(), reqVO);
            }
        }
        if (StrUtil.isBlank(userMessage)) {
            userMessage = "请执行任务。";
        }

        message.append(userMessage);

        if (!fullContext.isEmpty()) {
            message.append("\n</user_request>");
        }

        return message.toString();
    }

    /**
     * 构建上下文
     */
    private Map<String, Object> buildContext(AiChatReqVO reqVO) {
        Map<String, Object> context = new LinkedHashMap<>();

        // 1. 当前页面上下文
        String pageContext = buildCurrentPageContextPrompt(reqVO);
        if (StrUtil.isNotBlank(pageContext)) {
            context.put("page_context", pageContext);
        }

        // 2. 请求中已有的 context
        if (reqVO.getContext() != null) {
            context.putAll(reqVO.getContext());
        }

        // 3. 用户手动引用
        if (CollUtil.isNotEmpty(reqVO.getReferences())) {
            for (AiReferenceVO ref : reqVO.getReferences()) {
                try {
                    String key = ref.getTitle() != null ? ref.getTitle() : ref.getType() + "_" + ref.getId();

                    String fullText = null;
                    Integer startLine = null;
                    Integer endLine = null;
                    if (StrUtil.isNotBlank(ref.getMetadata())) {
                        try {
                            JSONObject metadataJson = JSONUtil.parseObj(ref.getMetadata());
                            fullText = metadataJson.getStr("fullText");
                            startLine = metadataJson.getInt("startLine");
                            endLine = metadataJson.getInt("endLine");
                        } catch (Exception e) {
                            log.warn("解析引用元数据失败: metadata={}", ref.getMetadata(), e);
                        }
                    }

                    StringBuilder contentBuilder = new StringBuilder();
                    contentBuilder.append("<meta>\n");
                    contentBuilder.append("  <type>").append(ref.getType()).append("</type>\n");
                    contentBuilder.append("  <id>").append(ref.getId()).append("</id>\n");

                    if (StrUtil.isNotBlank(fullText)) {
                        contentBuilder.append("  <line_range start=\"").append(startLine)
                                .append("\" end=\"").append(endLine).append("\" />\n");
                        contentBuilder.append("</meta>\n");
                        contentBuilder.append("<content>\n").append(fullText).append("\n</content>");
                    } else {
                        contentBuilder.append("  <hint>请使用查询工具获取完整内容</hint>\n");
                        contentBuilder.append("</meta>");
                    }

                    context.put(key, contentBuilder.toString());
                } catch (Exception e) {
                    log.warn("处理引用失败: type={}, id={}", ref.getType(), ref.getId(), e);
                }
            }
        }

        return context;
    }

    private String buildCurrentPageContextPrompt(AiChatReqVO reqVO) {
        List<AiReferenceVO> autoRefs = reqVO.getAutoReferences();
        if (CollUtil.isEmpty(autoRefs)) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("<page_context>\n");
        if (reqVO.getProjectId() != null && reqVO.getProjectId() > 0) {
            sb.append("  <project_id>").append(reqVO.getProjectId()).append("</project_id>\n");
        }

        for (AiReferenceVO autoRef : autoRefs) {
            String type = autoRef.getType();
            Long id = autoRef.getId();

            if (id != null && id > 0) {
                switch (type) {
                    case "script" -> sb.append("  <script_id>").append(id).append("</script_id>\n");
                    case "storyboard" ->
                        sb.append("  <storyboard_id>").append(id).append("</storyboard_id>\n");
                    case "asset", "character", "scene" -> {
                        sb.append("  <asset_id>").append(id).append("</asset_id>\n");
                        sb.append("  <asset_type>").append(type).append("</asset_type>\n");
                    }
                    case "project" -> {
                        /* projectId 已在上面设置 */ }
                    default -> sb.append("  <").append(type).append("_id>").append(id)
                            .append("</").append(type).append("_id>\n");
                }
            }
        }

        sb.append("</page_context>");
        return sb.toString();
    }

    private boolean isCancelled(String conversationId) {
        if (StrUtil.isBlank(conversationId)) {
            return false;
        }
        return Boolean.TRUE.equals(stringRedisTemplate.hasKey(CANCEL_FLAG_KEY + conversationId));
    }

    private void clearCancelFlag(String conversationId) {
        if (StrUtil.isNotBlank(conversationId)) {
            stringRedisTemplate.delete(CANCEL_FLAG_KEY + conversationId);
        }
    }
}
