package com.stonewu.fusion.service.ai.run;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AgentStateSessionIdsTests {

    @Test
    void assignsStableConversationAndImmutableExecutionGenerations() {
        assertThat(AgentStateSessionIds.conversation("conversation-7", "assistant"))
                .isEqualTo("afv:v2:conversation-7:assistant");

        String first = AgentStateSessionIds.recoveryGeneration(
                "conversation-7", "assistant", "run-1");
        String same = AgentStateSessionIds.recoveryGeneration(
                "conversation-7", "assistant", "run-1");
        String second = AgentStateSessionIds.recoveryGeneration(
                "conversation-7", "assistant", "run-2");

        assertThat(first)
                .isEqualTo(same)
                .startsWith("afv-root:")
                .hasSize(73)
                .isNotEqualTo(second);
    }

    @Test
    void rejectsAmbiguousSessionComponents() {
        assertThatThrownBy(() -> AgentStateSessionIds.recoveryGeneration(
                "conversation-7", "assistant", "bad\0run"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("runId");
    }
}
