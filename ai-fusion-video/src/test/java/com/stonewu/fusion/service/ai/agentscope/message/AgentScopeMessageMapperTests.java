package com.stonewu.fusion.service.ai.agentscope.message;

import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.UserMessage;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AgentScopeMessageMapperTests {

    @Test
    void mapsTextToStrongUserMessageWithoutRoleMutation() {
        AgentScopeMessageMapper mapper = new AgentScopeMessageMapper();

        UserMessage message = mapper.toUserMessage(" hello ");

        assertThat(message.getContent()).singleElement()
                .isInstanceOfSatisfying(TextBlock.class,
                        block -> assertThat(block.getText()).isEqualTo(" hello "));
        assertThat(mapper.toUserMessages("hello")).singleElement()
                .isInstanceOfSatisfying(UserMessage.class, mapped ->
                        assertThat(mapped.getContent()).singleElement()
                                .isInstanceOfSatisfying(TextBlock.class,
                                        block -> assertThat(block.getText()).isEqualTo("hello")));
        assertThatThrownBy(() -> mapper.toUserMessage("  "))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
