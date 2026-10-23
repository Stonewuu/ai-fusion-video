package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.repository.ai.AgentRunRepository;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.agentscope.context.AgentScopeRuntimeContextRequest;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.run.model.ResumedAgentRun;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScope;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScopeService;
import org.junit.jupiter.api.Test;
import reactor.core.scheduler.Schedulers;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CanvasExecutionPermissionTests {
    @Test
    void restoredCanvasNodeUsesAutomaticApprovalWithoutChangingTheRecordedKernel() {
        var request = restored("node-run");
        assertThat(request.toolPermission().mode()).isEqualTo(ToolExecutionMode.FULL_ACCESS);
        assertThat(request.toolExecution().canvasScope().nodeRunId()).isEqualTo("node-run");
    }

    @Test
    void interactiveAssistantKeepsTheUsersSelectedApprovalMode() {
        assertThat(restored(null).toolPermission().mode()).isEqualTo(ToolExecutionMode.DEFAULT);
    }

    private AgentScopeRuntimeContextRequest restored(String nodeRunId) {
        var repository = mock(AgentRunRepository.class);
        var schedulers = mock(AgentRuntimeSchedulers.class);
        var scopes = mock(CanvasAgentScopeService.class);
        Instant deadline = Instant.parse("2026-10-16T10:00:00Z");
        var persisted = AgentRun.builder().runId("run").conversationId("conversation").userId(1L).projectId(7L)
                .agentType("test").agentStateSessionId("session").canvasScopeJson("scope")
                .deadlineAt(LocalDateTime.ofInstant(deadline, ZoneOffset.UTC)).build();
        when(repository.findRun("run")).thenReturn(persisted);
        when(schedulers.journal()).thenReturn(Schedulers.immediate());
        when(scopes.decode("scope")).thenReturn(new CanvasAgentScope("canvas", 7L, List.of(), List.of(), null, 0, 0, 0, nodeRunId));
        var resumed = mock(ResumedAgentRun.class);
        when(resumed.runId()).thenReturn("run");
        when(resumed.deadline()).thenReturn(deadline);
        when(resumed.newOwnerInstanceId()).thenReturn("owner");
        when(resumed.newOwnerEpoch()).thenReturn(1L);
        return new AgentExecutionRuntimeContextRequests(repository, schedulers, scopes)
                .forResume(resumed, "test", ToolExecutionMode.DEFAULT).block();
    }
}
