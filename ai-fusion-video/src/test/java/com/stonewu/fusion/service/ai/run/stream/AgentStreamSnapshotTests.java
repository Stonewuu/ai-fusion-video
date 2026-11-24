package com.stonewu.fusion.service.ai.run.stream;

import com.stonewu.fusion.controller.ai.vo.AiChatStreamRespVO;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.service.ai.run.stream.AgentStreamHistoryRepository.History;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class AgentStreamSnapshotTests {
    @Test
    void committedMessageSuppressesLateRedisPublicationWithoutSuppressingAnotherChild() {
        String key = AgentStreamMessageRepository.key("child-a", "assistant", "reply");
        AgentStreamSnapshot snapshot = snapshot(List.of(AgentMessage.builder().runId("child-a")
                .role("assistant").projectionKey(key).content("完整回复").build()));
        assertThat(snapshot.includes(new AiChatStreamRespVO().setOutputType("CONTENT")
                .setSourceRunId("child-a").setMessageId(key).setContent("迟到片段"))).isFalse();
        assertThat(snapshot.includes(new AiChatStreamRespVO().setOutputType("CONTENT")
                .setSourceRunId("child-a").setMessageId(key).setContentMode("replace").setContent("完整回复"))).isFalse();
        assertThat(snapshot.includes(new AiChatStreamRespVO().setOutputType("CONTENT").setSourceRunId("child-b")
                .setMessageId(AgentStreamMessageRepository.key("child-b", "assistant", "reply")))).isTrue();
    }

    @Test
    void savedToolArgumentsKeepFollowingResultAndScopeDuplicateIdsToTheirRun() {
        AgentStreamSnapshot snapshot = snapshot(List.of(AgentMessage.builder().runId("child-a")
                .role("tool").toolCallId("lookup").toolStatus("running").build()));
        assertThat(snapshot.includes(tool("child-a", "TOOL_CALL"))).isFalse();
        assertThat(snapshot.includes(tool("child-a", "TOOL_FINISHED"))).isTrue();
        assertThat(snapshot.includes(tool("child-b", "TOOL_CALL"))).isTrue();
        AgentStreamSnapshot finished = snapshot(List.of(AgentMessage.builder().runId("child-a")
                .role("tool").toolCallId("lookup").toolStatus("success").build()));
        assertThat(finished.includes(tool("child-a", "TOOL_FINISHED"))).isFalse();
        assertThat(finished.includes(tool("child-b", "TOOL_FINISHED"))).isTrue();
    }

    private AgentStreamSnapshot snapshot(List<AgentMessage> messages) {
        AgentRun run = AgentRun.builder().runId("root").streamRunId("root")
                .deadlineAt(LocalDateTime.now().plusHours(1)).build();
        return new AgentStreamSnapshot(new History(run, messages, List.of()), 10);
    }

    private AiChatStreamRespVO tool(String run, String type) {
        return new AiChatStreamRespVO().setSourceRunId(run).setOutputType(type).setToolCallId("lookup");
    }
}
