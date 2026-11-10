package com.stonewu.fusion.service.ai.agentscope;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.config.ai.AiAgentRegistry;
import com.stonewu.fusion.enums.ai.AgentRunStatus;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.agentscope.context.ToolPermissionContext;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentCommand;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRun;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRunPort;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.tool.ToolCallParam;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentScopeSubAgentToolAdapterTests {

    @Test
    void storyboardReviewReceivesEpisodeIdentityAndSpecificRemainingWork() throws Exception {
        var definition = new AiAgentRegistry().getByType("script_to_storyboard").getSubAgentTools().stream()
                .filter(tool -> "episode_storyboard_writer".equals(tool.getToolName())).findFirst().orElseThrow();
        var mapper = new ObjectMapper();
        assertThat(mapper.readTree(definition.getParametersSchema()).path("properties").has("task")).isTrue();
        String task = "只读复核第七场剩余镜头的兵刃状态，不改写或重复生成已完成内容。";
        Map<String, Object> input = Map.of("scriptEpisodeId", 57, "task", task);
        var parent = mock(AgentKernelSpec.class);
        when(parent.promptVariables()).thenReturn(Map.of());
        var factory = mock(AgentKernelSpecFactory.class);
        when(factory.createChild(eq(parent), eq(definition), isNull(), eq(input)))
                .thenReturn(mock(AgentKernelSpec.class));
        var runs = mock(PlatformSubAgentRunPort.class);
        var child = new PlatformSubAgentRun("child", "parent", "call", definition.getToolName(),
                AgentRunStatus.COMPLETED, "已核对第七场", null);
        when(runs.start(any())).thenReturn(Mono.just(child));
        when(runs.awaitCompletion(child)).thenReturn(Mono.just(child));
        var lease = mock(RunLeaseGuard.class);
        when(lease.assertLease("parent", "owner", 1L)).thenReturn(Mono.empty());
        var adapter = new AgentScopeSubAgentToolAdapter(definition, parent,
                AgentScopeToolSchema.prepareSubAgent(mapper, definition.getParametersSchema(), definition.getToolName()),
                factory, () -> runs, lease, mapper);
        var runtime = RuntimeContext.builder().userId("42").sessionId("parent-session")
                .put(AgentRunContext.class, new AgentRunContext("parent", "owner", 1L, Instant.now().plusSeconds(30)))
                .put(CancellationContext.class, CancellationContext.noop())
                .put(ToolPermissionContext.class, new ToolPermissionContext(ToolExecutionMode.FULL_ACCESS)).build();

        adapter.callAsync(ToolCallParam.builder().runtimeContext(runtime).input(input)
                .toolUseBlock(ToolUseBlock.builder().id("call").name(definition.getToolName()).input(input).build())
                .build()).block();

        var captured = ArgumentCaptor.forClass(PlatformSubAgentCommand.class);
        verify(runs).start(captured.capture());
        assertThat(captured.getValue().messages()).hasSize(1);
        var delivered = mapper.readTree(captured.getValue().messages().getFirst().getTextContent());
        assertThat(delivered.path("scriptEpisodeId").asInt()).isEqualTo(57);
        assertThat(delivered.path("task").asText()).isEqualTo(task);
    }

    @Test
    void durableChildRunToolAllowsParallelInvocations() {
        AiAgentDefinition.SubAgentToolDef definition =
                AiAgentDefinition.SubAgentToolDef.builder()
                        .toolName("episode_scene_writer")
                        .description("Parse one episode")
                        .parametersSchema("{\"type\":\"object\"}")
                        .refAgentType("episode_scene_writer")
                        .build();
        ObjectMapper objectMapper = new ObjectMapper();

        AgentScopeSubAgentToolAdapter adapter = new AgentScopeSubAgentToolAdapter(
                definition,
                mock(AgentKernelSpec.class),
                AgentScopeToolSchema.prepareSubAgent(
                        objectMapper,
                        definition.getParametersSchema(),
                        definition.getToolName()),
                mock(AgentKernelSpecFactory.class),
                () -> mock(PlatformSubAgentRunPort.class),
                mock(RunLeaseGuard.class),
                objectMapper);

        assertThat(adapter.isConcurrencySafe()).isTrue();
        assertThat(adapter.isReadOnly()).isFalse();
    }
}
