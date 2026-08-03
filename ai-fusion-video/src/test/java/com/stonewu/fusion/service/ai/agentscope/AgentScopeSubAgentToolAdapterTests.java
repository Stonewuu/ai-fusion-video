package com.stonewu.fusion.service.ai.agentscope;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.agentscope.tool.PlatformSubAgentRunPort;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class AgentScopeSubAgentToolAdapterTests {

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
