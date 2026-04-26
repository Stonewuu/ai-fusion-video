package com.stonewu.fusion.service.ai;

import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class AiStreamRedisServiceTests {

    @Test
    void mainAgentTerminalEventOnlyMatchesConversationLevelTerminals() {
        AiChatStreamRespVO mainDone = new AiChatStreamRespVO().setOutputType("DONE");
        AiChatStreamRespVO mainError = new AiChatStreamRespVO().setOutputType("ERROR");
        AiChatStreamRespVO mainCancelled = new AiChatStreamRespVO().setOutputType("CANCELLED");
        AiChatStreamRespVO subAgentError = new AiChatStreamRespVO()
                .setOutputType("ERROR")
                .setAgentName("episode_storyboard_writer")
                .setParentToolCallId("call_1");
        AiChatStreamRespVO unmappedSubAgentError = new AiChatStreamRespVO()
                .setOutputType("ERROR")
                .setAgentName("episode_storyboard_writer");

        assertThat(AiStreamRedisService.isMainAgentTerminalEvent(mainDone)).isTrue();
        assertThat(AiStreamRedisService.isMainAgentTerminalEvent(mainError)).isTrue();
        assertThat(AiStreamRedisService.isMainAgentTerminalEvent(mainCancelled)).isTrue();
        assertThat(AiStreamRedisService.isMainAgentTerminalEvent(subAgentError)).isFalse();
        assertThat(AiStreamRedisService.isMainAgentTerminalEvent(unmappedSubAgentError)).isFalse();
    }

    @Test
    void mainAgentErrorAndCancelledIgnoreSubAgentEvents() {
        AiChatStreamRespVO mainError = new AiChatStreamRespVO().setOutputType("ERROR");
        AiChatStreamRespVO mainCancelled = new AiChatStreamRespVO().setOutputType("CANCELLED");
        AiChatStreamRespVO subAgentError = new AiChatStreamRespVO()
                .setOutputType("ERROR")
                .setAgentName("episode_storyboard_writer")
                .setParentToolCallId("call_1");
        AiChatStreamRespVO subAgentCancelled = new AiChatStreamRespVO()
                .setOutputType("CANCELLED")
                .setAgentName("episode_storyboard_writer")
                .setParentToolCallId("call_1");

        assertThat(AiStreamRedisService.isMainAgentErrorEvent(mainError)).isTrue();
        assertThat(AiStreamRedisService.isMainAgentCancelledEvent(mainCancelled)).isTrue();
        assertThat(AiStreamRedisService.isMainAgentErrorEvent(subAgentError)).isFalse();
        assertThat(AiStreamRedisService.isMainAgentCancelledEvent(subAgentCancelled)).isFalse();
    }
}