package com.stonewu.fusion.service.ai;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * Agent 消息服务
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AgentMessageService {

    private final AgentMessageMapper messageMapper;

    public List<AgentMessage> listByConversation(String conversationId) {
        return messageMapper.selectList(new LambdaQueryWrapper<AgentMessage>()
                .eq(AgentMessage::getConversationId, conversationId)
                .orderByAsc(AgentMessage::getMessageOrder));
    }

    public AgentMessage saveUserMessage(String conversationId, String content, String referencesJson) {
        int nextOrder = messageMapper.findMaxMessageOrder(conversationId) + 1;
        AgentMessage message = AgentMessage.builder()
                .conversationId(conversationId)
                .role("user")
                .content(content)
                .referencesJson(referencesJson)
                .messageOrder(nextOrder)
                .build();
        messageMapper.insert(message);
        return message;
    }

    public AgentMessage saveAssistantMessage(String conversationId, String content,
                                             String reasoningContent, Long reasoningDurationMs) {
        boolean hasContent = StrUtil.isNotEmpty(content);
        boolean hasReasoning = StrUtil.isNotEmpty(reasoningContent);
        if (!hasContent && !hasReasoning) {
            return null;
        }
        int nextOrder = messageMapper.findMaxMessageOrder(conversationId) + 1;
        AgentMessage message = AgentMessage.builder()
                .conversationId(conversationId)
                .role("assistant")
                .content(hasContent ? content : null)
                .reasoningContent(hasReasoning ? reasoningContent : null)
                .reasoningDurationMs(reasoningDurationMs)
                .messageOrder(nextOrder)
                .build();
        messageMapper.insert(message);
        return message;
    }

    public AgentMessage saveToolCall(String conversationId, String toolName,
                                     String toolStatus, String content,
                                     String toolCallId, String parentToolCallId) {
        int nextOrder = messageMapper.findMaxMessageOrder(conversationId) + 1;
        AgentMessage message = AgentMessage.builder()
                .conversationId(conversationId)
                .role("tool")
                .toolName(toolName)
                .toolStatus(toolStatus)
                .content(content)
                .toolCallId(toolCallId)
                .parentToolCallId(parentToolCallId)
                .messageOrder(nextOrder)
                .build();
        messageMapper.insert(message);
        return message;
    }
}
