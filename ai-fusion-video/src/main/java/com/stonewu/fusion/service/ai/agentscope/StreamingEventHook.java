package com.stonewu.fusion.service.ai.agentscope;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import io.agentscope.core.agent.Agent;
import io.agentscope.core.hook.ActingChunkEvent;
import io.agentscope.core.hook.ErrorEvent;
import io.agentscope.core.hook.Hook;
import io.agentscope.core.hook.HookEvent;
import io.agentscope.core.hook.PostActingEvent;
import io.agentscope.core.hook.PostCallEvent;
import io.agentscope.core.hook.PostReasoningEvent;
import io.agentscope.core.hook.PreActingEvent;
import io.agentscope.core.hook.ReasoningChunkEvent;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ThinkingBlock;
import io.agentscope.core.message.ToolUseBlock;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * AgentScope 流式事件 Hook
 * <p>
 * 拦截 Agent 执行过程中的各种事件（推理、工具调用等），
 * 将其转化为 {@link AiChatStreamRespVO} 并推入 {@link Sinks.Many}，
 * 实现流式输出。
 * <p>
 * 核心能力：
 * 1. 捕获 ReasoningChunkEvent → 推送 REASONING / CONTENT 事件
 * 2. 捕获 PreActingEvent → 推送 TOOL_CALL 事件
 * 3. 捕获 PostActingEvent → 推送 TOOL_FINISHED 事件
 * 4. 子 Agent 事件自动识别与归属：
 * - 主 Agent 调用子 Agent 工具时，记录 toolCallId 到待分配队列
 * - 子 Agent 首次触发事件时，从队列中取出 toolCallId 建立映射
 * - 后续子 Agent 事件通过 agentKey 查找已建立的映射
 */
@Slf4j
public class StreamingEventHook implements Hook {

    private final Sinks.Many<AiChatStreamRespVO> eventSink;
    private final String conversationId;
    private final String messageId;
    private final String mainAgentName;
    private final AgentCancellationToken cancellationToken;

    /** 思考开始时间（按 Agent 实例 key 隔离） */
    private final ConcurrentHashMap<String, Long> reasoningStartTimes = new ConcurrentHashMap<>();

    /** 思考持续时间（按 Agent 实例 key 隔离），只记录一次 */
    private final ConcurrentHashMap<String, Long> reasoningDurations = new ConcurrentHashMap<>();

    /**
     * 已注册的子 Agent 工具名集合。
     * <p>
     * 通过 {@link #registerSubAgentToolName(String)} 注册，
     * 用于在 handlePreActing 中判断主 Agent 调用的工具是否是子 Agent 工具。
     */
    private final Set<String> subAgentToolNames = ConcurrentHashMap.newKeySet();

    /**
     * 待分配的子 Agent 调用队列：subAgentName → Queue<toolCallId>
     * <p>
     * 当主 Agent 调用子 Agent 工具时（PreActingEvent），将 toolCallId 入队。
     * 当子 Agent 首次触发事件时，出队分配给该实例。
     */
    private final ConcurrentHashMap<String, ConcurrentLinkedQueue<String>> pendingSubAgentCalls = new ConcurrentHashMap<>();

    /**
     * 已建立的子 Agent 实例映射：agentInstanceKey → parentToolCallId
     * <p>
     * agentInstanceKey 使用 Agent 对象的 identity hash code 区分同名并行实例。
     */
    private final ConcurrentHashMap<String, String> activeSubAgentCalls = new ConcurrentHashMap<>();

    /**
     * 当前会话中已登记的活跃 Agent 实例。
     * <p>
     * 用于在用户取消时，对主 Agent 和所有已创建的子 Agent 调用官方 interrupt()。
     */
    private final ConcurrentHashMap<String, Agent> activeAgents = new ConcurrentHashMap<>();

    public StreamingEventHook(Sinks.Many<AiChatStreamRespVO> eventSink,
            String conversationId,
            String messageId,
            String mainAgentName,
            AgentCancellationToken cancellationToken) {
        this.eventSink = eventSink;
        this.conversationId = conversationId;
        this.messageId = messageId;
        this.mainAgentName = mainAgentName;
        this.cancellationToken = cancellationToken;
    }

    @Override
    public <T extends HookEvent> Mono<T> onEvent(T event) {
        registerActiveAgent(event.getAgent());

        // 每个事件触发时检查取消标志，尽早中断 Agent 的 ReAct 循环
        if (cancellationToken.isCancelled()) {
            log.info("[StreamingEventHook] 检测到取消标志，中断 Agent 执行: event={}",
                    event.getClass().getSimpleName());
            return Mono.error(new AgentCancelledException("Agent 执行已被用户取消"));
        }

        try {
            if (event instanceof ReasoningChunkEvent e) {
                handleReasoningChunk(e);
            } else if (event instanceof PostReasoningEvent e) {
                handlePostReasoning(e);
            } else if (event instanceof PreActingEvent e) {
                handlePreActing(e);
            } else if (event instanceof PostActingEvent e) {
                handlePostActing(e);
            } else if (event instanceof ActingChunkEvent) {
                // 工具流式进度，暂不处理
            } else if (event instanceof PostCallEvent e) {
                handlePostCall(e);
            } else if (event instanceof ErrorEvent e) {
                handleError(e);
            }
        } catch (AgentCancelledException ex) {
            // 取消异常直接向上传播
            return Mono.error(ex);
        } catch (Exception ex) {
            log.error("[StreamingEventHook] 处理事件异常: eventType={}",
                    event.getClass().getSimpleName(), ex);
        }
        return Mono.just(event);
    }

    // ========== 事件处理 ==========

    /**
     * 推理流式块：模型生成过程中的每个增量 chunk
     */
    private void handleReasoningChunk(ReasoningChunkEvent event) {
        String agentName = event.getAgent().getName();
        String agentKey = getAgentKey(event);
        String parentCallId = resolveParentCallId(event);

        Msg incrementalChunk = event.getIncrementalChunk();
        if (incrementalChunk == null) {
            // log.debug("[StreamingEventHook] 收到event: {}, agent={}, parentCallId={}, chunk=null",
            //         event.getClass().getSimpleName(), agentName, parentCallId);
            return;
        }

        // log.debug("[StreamingEventHook] 收到event: {}, agent={}, parentCallId={}, textContent={}",
        //          event.getClass().getSimpleName(), agentName, parentCallId, incrementalChunk.getTextContent());

        for (ContentBlock block : incrementalChunk.getContent()) {
            if (block instanceof ThinkingBlock thinkingBlock) {
                String thinkingText = thinkingBlock.getThinking();
                if (thinkingText != null && !thinkingText.isEmpty()) {
                    emitReasoningEvent(agentName, agentKey, parentCallId, thinkingText, "ThinkingBlock");
                }
            } else if (block instanceof TextBlock textBlock) {
                String text = textBlock.getText();
                if (text != null && !text.isEmpty()) {
                    // 计算思考耗时（首次从 REASONING 切换到 CONTENT 时）
                    Long durationMs = null;
                    if (reasoningStartTimes.containsKey(agentKey)
                            && !reasoningDurations.containsKey(agentKey)) {
                        durationMs = System.currentTimeMillis() - reasoningStartTimes.get(agentKey);
                        reasoningDurations.put(agentKey, durationMs);
                    }

                    AiChatStreamRespVO resp = new AiChatStreamRespVO()
                            .setMessageId(messageId)
                            .setConversationId(conversationId)
                            .setOutputType("CONTENT")
                            .setContent(text)
                            .setParentToolCallId(parentCallId)
                            .setAgentName(isSubAgent(agentName) ? agentName : null)
                            .setFinished(false);
                    if (durationMs != null) {
                        resp.setReasoningDurationMs(durationMs);
                    }
                    emitEvent(resp);
                }
            }
        }
    }

    /**
     * 推理完成：LLM 生成结束，清理当前轮的状态
     */
    private void handlePostReasoning(PostReasoningEvent event) {
        String agentName = event.getAgent().getName();
        String agentKey = getAgentKey(event);
        String parentCallId = resolveParentCallId(event);

        emitReasoningDurationIfNeeded(agentName, agentKey, parentCallId);
        reasoningStartTimes.remove(agentKey);
        reasoningDurations.remove(agentKey);
    }

    /**
     * 工具执行前：发送 TOOL_CALL 事件
     * <p>
     * 如果是主 Agent 调用子 Agent 工具，同时记录 toolCallId 到待分配队列。
     */
    private void handlePreActing(PreActingEvent event) {
        String agentName = event.getAgent().getName();
        ToolUseBlock toolUse = event.getToolUse();
        String toolCallId = toolUse.getId();
        String toolName = toolUse.getName();
        String arguments = JSONUtil.toJsonStr(toolUse.getInput());
        String parentCallId = resolveParentCallId(event);

        log.info("[StreamingEventHook] 工具调用开始: agent={}, tool={}, callId={}",
                agentName, toolName, toolCallId);

        // 如果主 Agent 调用了子 Agent 工具，记录待分配映射
        if (isMainAgent(agentName) && subAgentToolNames.contains(toolName)) {
            pendingSubAgentCalls
                    .computeIfAbsent(toolName, k -> new ConcurrentLinkedQueue<>())
                    .offer(toolCallId);
            log.debug("[StreamingEventHook] 子Agent调用入队: toolName={}, callId={}", toolName, toolCallId);
        }

        emitEvent(new AiChatStreamRespVO()
                .setMessageId(messageId)
                .setConversationId(conversationId)
                .setOutputType("TOOL_CALL")
                .setToolCalls(List.of(new AiChatStreamRespVO.ToolCallVO()
                        .setId(toolCallId)
                        .setName(toolName)
                        .setArguments(arguments)))
                .setParentToolCallId(parentCallId)
                .setAgentName(isSubAgent(agentName) ? agentName : null)
                .setFinished(false));
    }

    /**
     * 工具执行后：发送 TOOL_FINISHED 事件
     */
    private void handlePostActing(PostActingEvent event) {
        String agentName = event.getAgent().getName();
        String toolCallId = event.getToolResult().getId();
        String toolName = event.getToolResult().getName();
        String parentCallId = resolveParentCallId(event);

        // 提取工具结果文本
        String resultText = event.getToolResult().getOutput().stream()
                .filter(block -> block instanceof TextBlock)
                .map(block -> ((TextBlock) block).getText())
                .findFirst()
                .orElse("");

        // 去除 AgentScope SubAgentTool 自动附加的 "session_id: xxx" 行
        resultText = stripSessionId(resultText);

        String toolStatus = detectToolStatus(resultText);

        log.info("[StreamingEventHook] 工具调用完成: agent={}, tool={}, status={}",
                agentName, toolName, toolStatus);

        emitEvent(new AiChatStreamRespVO()
                .setMessageId(messageId)
                .setConversationId(conversationId)
                .setOutputType("TOOL_FINISHED")
                .setToolCallId(toolCallId)
                .setToolName(toolName)
                .setToolResult(resultText)
                .setToolStatus(toolStatus)
                .setParentToolCallId(parentCallId)
                .setAgentName(isSubAgent(agentName) ? agentName : null)
                .setFinished(false));
    }

    /**
     * Agent 调用完成
     */
    private void handlePostCall(PostCallEvent event) {
        String agentName = event.getAgent().getName();
        String agentKey = getAgentKey(event);

        activeAgents.remove(agentKey);

        if (isMainAgent(agentName)) {
            Msg finalMsg = event.getFinalMessage();
            if (finalMsg != null) {
                String finalContent = finalMsg.getTextContent();
                if (finalContent != null && !finalContent.isEmpty()) {
                    log.debug("[StreamingEventHook] 主 Agent 完成: contentLength={}",
                            finalContent.length());
                }
            }
        } else {
            // 子 Agent 完成，清理映射
            activeSubAgentCalls.remove(agentKey);
            log.debug("[StreamingEventHook] 子 Agent 完成，清理映射: agentKey={}", agentKey);
        }
    }

    /**
     * 错误处理
     */
    private void handleError(ErrorEvent event) {
        String agentName = event.getAgent().getName();
        activeAgents.remove(getAgentKey(event));
        String errorMsg = event.getError() != null ? event.getError().getMessage() : "未知错误";
        log.error("[StreamingEventHook] Agent 错误: agent={}, error={}", agentName, errorMsg);

        emitEvent(new AiChatStreamRespVO()
                .setMessageId(messageId)
                .setConversationId(conversationId)
                .setOutputType("ERROR")
                .setError(agentName + " 执行出错: " + errorMsg)
                .setAgentName(isSubAgent(agentName) ? agentName : null)
                .setFinished(true));
    }

    // ========== 子 Agent 映射 ==========

    /**
     * 注册子 Agent 工具名
     * <p>
     * 在构建 Toolkit 时调用，告知 Hook 哪些工具名是子 Agent 工具，
     * 以便在 handlePreActing 中自动建立 parentToolCallId 映射。
     */
    public void registerSubAgentToolName(String toolName) {
        subAgentToolNames.add(toolName);
        log.debug("[StreamingEventHook] 注册子Agent工具名: {}", toolName);
    }

    /**
     * 登记当前执行中的 Agent 实例。
     */
    public void registerActiveAgent(Agent agent) {
        if (agent == null) {
            return;
        }
        activeAgents.put(getAgentKey(agent), agent);
    }

    /**
     * 对当前会话中所有已登记的 Agent 实例发送官方 interrupt 信号。
     */
    public void interruptTrackedAgents() {
        activeAgents.forEach((agentKey, agent) -> {
            try {
                agent.interrupt();
                log.info("[StreamingEventHook] 已发送 interrupt 信号: agentKey={}, agentName={}",
                        agentKey, agent.getName());
            } catch (Exception e) {
                log.warn("[StreamingEventHook] 发送 interrupt 信号失败: agentKey={}, agentName={}",
                        agentKey, agent.getName(), e);
            }
        });
    }

    /**
     * 清空当前会话中登记的 Agent 实例。
     */
    public void clearTrackedAgents() {
        activeAgents.clear();
    }

    // ========== 内部方法 ==========

    private String getAgentKey(Agent agent) {
        return agent.getName() + ":" + System.identityHashCode(agent);
    }

    private String getAgentKey(HookEvent event) {
        return getAgentKey(event.getAgent());
    }

    private boolean isMainAgent(String agentName) {
        return mainAgentName.equals(agentName);
    }

    private boolean isSubAgent(String agentName) {
        return !isMainAgent(agentName);
    }

    /**
     * 解析事件对应的 parentToolCallId
     * <p>
     * 对于主 Agent 事件，返回 null。
     * 对于子 Agent 事件：
     * 1. 先检查 activeSubAgentCalls 是否已有映射（已分配过）
     * 2. 如果没有，尝试从 pendingSubAgentCalls 中按 agentName 出队分配
     */
    private String resolveParentCallId(HookEvent event) {
        String agentName = event.getAgent().getName();
        if (isMainAgent(agentName)) {
            return null;
        }

        String agentKey = getAgentKey(event);

        // 1. 已有映射：直接返回
        String existing = activeSubAgentCalls.get(agentKey);
        if (existing != null) {
            return existing;
        }

        // 2. 尝试从待分配队列中取出 toolCallId 并建立映射
        ConcurrentLinkedQueue<String> queue = pendingSubAgentCalls.get(agentName);
        if (queue != null) {
            String toolCallId = queue.poll();
            if (toolCallId != null) {
                activeSubAgentCalls.put(agentKey, toolCallId);
                log.info("[StreamingEventHook] 子Agent映射建立: agentKey={} -> parentCallId={}",
                        agentKey, toolCallId);
                return toolCallId;
            }
        }

        log.warn("[StreamingEventHook] 子Agent未找到parentCallId: agentName={}, agentKey={}",
                agentName, agentKey);
        return null;
    }

    private synchronized void emitEvent(AiChatStreamRespVO event) {
        Sinks.EmitResult result = eventSink.tryEmitNext(event);
        if (result.isFailure()) {
            log.warn("[StreamingEventHook] 事件发送失败: result={}, type={}, agentName={}, content={}, currentThread={}",
                    result, event.getOutputType(), event.getAgentName(),
                    event.getContent() != null ? event.getContent().substring(0, Math.min(event.getContent().length(), 100)) : null,
                    Thread.currentThread().getName());
        }
    }

    private void emitReasoningEvent(String agentName, String agentKey, String parentCallId,
            String reasoningText, String sourceType) {
        reasoningStartTimes.putIfAbsent(agentKey, System.currentTimeMillis());
        log.debug("[StreamingEventHook] 思考增量: agent={}, parentCallId={}, sourceType={}, content={}",
                agentName, parentCallId, sourceType, reasoningText);

        emitEvent(new AiChatStreamRespVO()
                .setMessageId(messageId)
                .setConversationId(conversationId)
                .setOutputType("REASONING")
                .setReasoningContent(reasoningText)
                .setReasoningStartTime(reasoningStartTimes.get(agentKey))
                .setParentToolCallId(parentCallId)
                .setAgentName(isSubAgent(agentName) ? agentName : null)
                .setFinished(false));
    }

    private void emitReasoningDurationIfNeeded(String agentName, String agentKey, String parentCallId) {
        Long startTime = reasoningStartTimes.get(agentKey);
        if (startTime == null || reasoningDurations.containsKey(agentKey)) {
            return;
        }

        Long durationMs = System.currentTimeMillis() - startTime;
        reasoningDurations.put(agentKey, durationMs);

        emitEvent(new AiChatStreamRespVO()
                .setMessageId(messageId)
                .setConversationId(conversationId)
                .setOutputType("CONTENT")
                .setReasoningDurationMs(durationMs)
                .setParentToolCallId(parentCallId)
                .setAgentName(isSubAgent(agentName) ? agentName : null)
                .setFinished(false));
    }

    /**
     * 根据工具返回内容检测执行状态
     */
    private String detectToolStatus(String toolResult) {
        if (toolResult == null || toolResult.isBlank()) {
            return "success";
        }
        try {
            if (toolResult.trim().startsWith("{")) {
                JSONObject json = JSONUtil.parseObj(toolResult);
                String status = json.getStr("status");
                if ("error".equals(status) || "not_implemented".equals(status)) {
                    return "error";
                }
            }
        } catch (Exception ignored) {
        }
        String lower = toolResult.toLowerCase();
        if (lower.contains("工具执行失败") || lower.contains("执行异常")) {
            return "error";
        }
        return "success";
    }

    /**
     * 去除 AgentScope SubAgentTool 自动附加的 "session_id: xxx" 行
     * <p>
     * SubAgentTool 的工具结果格式为：
     * "session_id: xxxxxxx\n\n{实际内容}"
     * 需要将 session_id 行及其后的空行移除，只保留实际内容。
     */
    private String stripSessionId(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }
        // 匹配行首的 "session_id: xxx" 及后续空行
        return text.replaceAll("(?m)^session_id:.*\\n*", "").trim();
    }
}
