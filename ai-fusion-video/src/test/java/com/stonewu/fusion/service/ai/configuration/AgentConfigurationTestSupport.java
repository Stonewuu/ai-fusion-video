package com.stonewu.fusion.service.ai.configuration;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.config.ai.AiAgentRegistry;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentScopeSkillRegistry;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentUserSkillService;

import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public final class AgentConfigurationTestSupport {
    private AgentConfigurationTestSupport() { }
    public static AiAgentService agents(AiAgentRegistry registry) {
        var repository = mock(AgentUserOverrideRepository.class);
        when(repository.get(anyLong(), anyString())).thenReturn(new AgentUserOverrideRepository.Stored(0, "{}"));
        return new AiAgentService(registry, new AgentDefinitionConfiguration(registry, repository, new ObjectMapper(), new AgentScopeV2Properties()));
    }
    public static AgentKernelSkills skills(AiAgentService agents) {
        return new AgentKernelSkills(new AgentSkillSelection(mock(AgentScopeSkillRegistry.class), mock(AgentUserSkillService.class)), agents, new ObjectMapper());
    }
}
