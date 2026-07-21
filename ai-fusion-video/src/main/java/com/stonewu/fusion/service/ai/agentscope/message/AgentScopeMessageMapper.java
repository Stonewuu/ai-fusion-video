package com.stonewu.fusion.service.ai.agentscope.message;

import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.UserMessage;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public final class AgentScopeMessageMapper {

    public UserMessage toUserMessage(String text) {
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("text must not be blank");
        }
        return new UserMessage(List.of(TextBlock.builder().text(text).build()));
    }

    public List<Msg> toUserMessages(String text) {
        return List.of(toUserMessage(text));
    }
}
