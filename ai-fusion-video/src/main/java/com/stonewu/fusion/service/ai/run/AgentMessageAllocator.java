package com.stonewu.fusion.service.ai.run;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Objects;

/**
 * Serializes message-order allocation on the owning conversation row.
 */
@Service
@RequiredArgsConstructor
public class AgentMessageAllocator {

    private final AgentConversationMapper conversationMapper;
    private final AgentMessageMapper messageMapper;

    @Transactional
    public long append(String conversationId, AgentMessage message) {
        if (StrUtil.isBlank(conversationId)) {
            throw new IllegalArgumentException("conversationId must not be blank");
        }
        Objects.requireNonNull(message, "message must not be null");

        AgentConversation conversation =
                conversationMapper.selectByConversationIdForUpdate(conversationId);
        if (conversation == null) {
            throw new IllegalStateException("Agent conversation does not exist: " + conversationId);
        }

        Long nextMessageOrder = conversation.getNextMessageOrder();
        Integer messageCount = conversation.getMessageCount();
        if (nextMessageOrder == null || nextMessageOrder < 1) {
            throw new IllegalStateException(
                    "Agent conversation has an invalid next message order: " + conversationId);
        }
        if (messageCount == null || messageCount < 0) {
            throw new IllegalStateException(
                    "Agent conversation has an invalid message count: " + conversationId);
        }

        long order = nextMessageOrder;
        message.setConversationId(conversationId);
        message.setMessageOrder(order);
        if (messageMapper.insert(message) != 1) {
            throw new IllegalStateException("Agent message insert did not affect exactly one row");
        }

        conversation.setNextMessageOrder(order + 1);
        conversation.setMessageCount(messageCount + 1);
        conversation.setLastMessageTime(LocalDateTime.now());
        if (conversationMapper.updateById(conversation) != 1) {
            throw new IllegalStateException("Agent conversation counter update did not affect exactly one row");
        }
        return order;
    }
}
