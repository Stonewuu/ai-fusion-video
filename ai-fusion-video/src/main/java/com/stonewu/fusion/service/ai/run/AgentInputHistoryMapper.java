package com.stonewu.fusion.service.ai.run;

import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;

@Component
public final class AgentInputHistoryMapper {

    public String userContent(List<Msg> messages) {
        List<Msg> safeMessages = List.copyOf(Objects.requireNonNull(
                messages, "messages must not be null"));
        String content = safeMessages.stream()
                .filter(message -> message.getRole() == MsgRole.USER)
                .map(Msg::getTextContent)
                .filter(text -> text != null && !text.isBlank())
                .reduce((ignored, latest) -> latest)
                .orElseThrow(() -> new IllegalArgumentException(
                        "messages must contain textual input for durable admission"));
        return content;
    }
}
