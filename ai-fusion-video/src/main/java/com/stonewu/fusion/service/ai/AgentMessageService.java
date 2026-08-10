package com.stonewu.fusion.service.ai;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.service.ai.run.AgentMessageAllocator;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Objects;

/**
 * Agent 消息服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AgentMessageService {

    private final AgentMessageMapper messageMapper;
    private final AgentMessageAllocator messageAllocator;

    public List<AgentMessage> listByConversation(String conversationId) {
        return messageMapper.selectList(new LambdaQueryWrapper<AgentMessage>()
                .eq(AgentMessage::getConversationId, conversationId)
                .orderByAsc(AgentMessage::getMessageOrder));
    }

    public AgentMessage saveUserMessage(String conversationId, String content, String referencesJson) {
        AgentMessage message = AgentMessage.builder()
                .role("user")
                .content(content)
                .referencesJson(referencesJson)
                .build();
        messageAllocator.append(conversationId, message);
        return message;
    }

    public AgentMessage saveAssistantMessage(String conversationId, String content,
                                             String reasoningContent, Long reasoningDurationMs) {
        return saveAssistantMessage(conversationId, content, reasoningContent,
                reasoningDurationMs, null);
    }

    public AgentMessage saveAssistantMessage(String conversationId, String content,
                                             String reasoningContent, Long reasoningDurationMs,
                                             String parentToolCallId) {
        boolean hasContent = StrUtil.isNotEmpty(content);
        boolean hasReasoning = StrUtil.isNotEmpty(reasoningContent);
        if (!hasContent && !hasReasoning) {
            return null;
        }
        AgentMessage message = AgentMessage.builder()
                .role("assistant")
                .content(hasContent ? content : null)
                .parentToolCallId(parentToolCallId)
                .reasoningContent(hasReasoning ? reasoningContent : null)
                .reasoningDurationMs(reasoningDurationMs)
                .build();
        messageAllocator.append(conversationId, message);
        return message;
    }

    public AgentMessage saveToolCall(String conversationId, String toolName,
                                     String toolStatus, String content,
                                     String toolCallId, String parentToolCallId) {
        AgentMessage message = AgentMessage.builder()
                .role("tool")
                .toolName(toolName)
                .toolStatus(toolStatus)
                .content(content)
                .toolCallId(toolCallId)
                .parentToolCallId(parentToolCallId)
                .build();
        messageAllocator.append(conversationId, message);
        return message;
    }

    /** Persists an already validated event projection through the shared order allocator. */
    public AgentMessage saveProjectedMessage(
            String conversationId, AgentMessage projection) {
        AgentMessage message = Objects.requireNonNull(
                projection, "projection must not be null");
        if (StrUtil.isBlank(message.getRunId())
                || StrUtil.isBlank(message.getProjectionKey())
                || StrUtil.isBlank(message.getRole())) {
            throw new IllegalArgumentException(
                    "Projected message requires runId, projectionKey, and role");
        }
        messageAllocator.append(conversationId, message);
        return message;
    }
}
