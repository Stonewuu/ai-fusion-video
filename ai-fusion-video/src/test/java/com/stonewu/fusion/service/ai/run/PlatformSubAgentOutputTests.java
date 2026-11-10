package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRun;
import com.stonewu.fusion.service.ai.run.kernel.AgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import com.stonewu.fusion.service.ai.run.model.CommittedAgentEvent;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PlatformSubAgentOutputTests {
    @Test
    void childOutputRetainsWhitespaceAndDoesNotFinishOnADescendantsMirroredTerminal() {
        var replay = mock(AgentRunReplayService.class);
        var service = new PlatformSubAgentRunService(mock(AgentRunCoordinator.class), mock(AgentKernelSnapshotBuilder.class),
                mock(AgentInputHistoryMapper.class), mock(AgentExecutionRuntimeContextRequests.class), mock(RunExecutionSupervisor.class),
                replay, mock(AgentMessageProjectionService.class), mock(CancellationCoordinator.class),
                mock(AgentRuntimeInstanceIdentity.class), new AgentScopeV2Properties());
        when(replay.replayThenLive("child", 0)).thenReturn(Flux.just(
                event(1, "CONTENT", "甲", false), event(2, "CONTENT", " \n", false),
                event(3, "DONE", "孙任务已经结束", true), event(4, "CONTENT", "乙", false), event(5, "DONE", "", false)));
        var result = service.awaitCompletion(new PlatformSubAgentRun("child", "parent", "tool", "writer", AgentRunStatus.RUNNING))
                .block(Duration.ofSeconds(1));
        assertThat(result).isNotNull();
        assertThat(result.status()).isEqualTo(AgentRunStatus.COMPLETED);
        assertThat(result.result()).isEqualTo("甲 \n乙");
    }

    @Test
    void modelRetryDoesNotSendTheFailedPartialAnswerBackToTheParent() {
        var replay = mock(AgentRunReplayService.class);
        var service = new PlatformSubAgentRunService(mock(AgentRunCoordinator.class), mock(AgentKernelSnapshotBuilder.class),
                mock(AgentInputHistoryMapper.class), mock(AgentExecutionRuntimeContextRequests.class), mock(RunExecutionSupervisor.class),
                replay, mock(AgentMessageProjectionService.class), mock(CancellationCoordinator.class),
                mock(AgentRuntimeInstanceIdentity.class), new AgentScopeV2Properties());
        when(replay.replayThenLive("child", 0)).thenReturn(Flux.just(
                replyEvent(1, "CONTENT", "已经保存。", "good"), replyEvent(2, "CONTENT", "残缺回答", "failed"),
                replyEvent(3, "MODEL_RETRY", "正在重试", "failed"), replyEvent(4, "CONTENT", "已完成。", "retry"),
                event(5, "DONE", "", false)));
        var result = service.awaitCompletion(new PlatformSubAgentRun("child", "parent", "tool", "writer", AgentRunStatus.RUNNING))
                .block(Duration.ofSeconds(1));
        assertThat(result.result()).isEqualTo("已经保存。已完成。");
    }

    private CommittedAgentEvent replyEvent(long sequence, String type, String text, String replyId) {
        var payload = JsonNodeFactory.instance.objectNode().put("delta", text);
        return new CommittedAgentEvent(sequence, "child", sequence, new AgentEventEnvelope("event-" + sequence,
                type, "main", replyId, null, null, null, null, type, payload, Instant.now()), Instant.now());
    }

    private CommittedAgentEvent event(long sequence, String type, String text, boolean mirrored) {
        var payload = JsonNodeFactory.instance.objectNode().put("delta", text).put("_platformMirroredChildEvent", mirrored);
        return new CommittedAgentEvent(sequence, "child", sequence, new AgentEventEnvelope("event-" + sequence,
                type, "main", null, null, null, null, null, type, payload, Instant.now()), Instant.now());
    }
}
