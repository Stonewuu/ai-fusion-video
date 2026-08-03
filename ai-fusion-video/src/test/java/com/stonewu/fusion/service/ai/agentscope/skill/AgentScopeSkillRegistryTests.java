package com.stonewu.fusion.service.ai.agentscope.skill;

import com.stonewu.fusion.config.AgentScopeV2Properties;
import io.agentscope.core.skill.AgentSkill;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class AgentScopeSkillRegistryTests {

    @Test
    void acceptsClasspathRepositoryWithoutSkills() {
        AgentScopeV2Properties properties = new AgentScopeV2Properties();
        properties.getSkills().setEnabled(true);
        AgentScopeV2Properties.SkillRepository empty =
                new AgentScopeV2Properties.SkillRepository();
        empty.setLocation("classpath:agentscope/empty-skills");
        properties.getSkills().setRepositories(Map.of("empty", empty));

        AgentScopeSkillRegistry registry = new AgentScopeSkillRegistry(properties);
        try {
            assertThat(registry.enabled()).isTrue();
            assertThat(registry.repositories()).hasSize(1);
            assertThat(registry.skills()).isEmpty();
            assertThat(registry.catalog()).isEmpty();
        } finally {
            registry.destroy();
        }
    }

    @Test
    void loadsBundledSkillRepositoryFromConfiguration() {
        AgentScopeV2Properties properties = new AgentScopeV2Properties();
        properties.getSkills().setEnabled(true);
        AgentScopeV2Properties.SkillRepository bundled =
                new AgentScopeV2Properties.SkillRepository();
        bundled.setLocation("classpath:agentscope/skills");
        properties.getSkills().setRepositories(Map.of("bundled", bundled));
        properties.getSkills().setDisplayNames(Map.of(
                "test-skill", "测试技能"));

        AgentScopeSkillRegistry registry = new AgentScopeSkillRegistry(properties);
        try {
            assertThat(registry.enabled()).isTrue();
            assertThat(registry.repositories()).hasSize(1);
            AgentSkill skill = registry.repositories().getFirst()
                    .getSkill("test-skill");
            assertThat(skill.getName()).isEqualTo("test-skill");
            assertThat(skill.getDescription()).contains("测试");
            assertThat(skill.getSource()).isEqualTo("bundled");
            assertThat(registry.catalog()).singleElement().satisfies(reference -> {
                assertThat(reference.name()).isEqualTo("test-skill");
                assertThat(reference.displayName()).isEqualTo("测试技能");
            });
        } finally {
            registry.destroy();
        }
    }
}
