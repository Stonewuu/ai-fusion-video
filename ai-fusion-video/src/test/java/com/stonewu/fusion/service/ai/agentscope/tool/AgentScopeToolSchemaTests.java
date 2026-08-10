package com.stonewu.fusion.service.ai.agentscope.tool;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.config.ai.AiAgentRegistry;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AgentScopeToolSchemaTests {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void normalizesObjectSchemasToRejectUnknownInput() {
        AgentScopeToolSchema.PreparedSchema schema = AgentScopeToolSchema.prepare(
                objectMapper,
                "{\"type\":\"object\",\"properties\":{}}",
                "query");

        assertThat(schema.value())
                .containsEntry("type", "object")
                .containsEntry("additionalProperties", false);
        assertThat(schema.canonicalJson()).contains("\"additionalProperties\":false");
    }

    @Test
    void rejectsMissingMalformedAndPermissiveSchemasWithoutAlternativeBehavior() {
        assertThatThrownBy(() -> AgentScopeToolSchema.prepare(
                objectMapper, null, "query"))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> AgentScopeToolSchema.prepare(
                objectMapper, "[]", "query"))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> AgentScopeToolSchema.prepare(
                objectMapper,
                "{\"type\":\"object\",\"additionalProperties\":true}",
                "query"))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsArraySchemasWithoutAnItemDefinition() {
        assertThatThrownBy(() -> AgentScopeToolSchema.prepare(
                objectMapper,
                "{\"type\":\"object\",\"properties\":{\"values\":{\"type\":\"array\"}}}",
                "query"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must declare items")
                .hasMessageContaining("$.properties.values");
    }

    @Test
    void rejectsMissingSubAgentSchemaWithoutCreatingADefault() {
        assertThatThrownBy(() -> AgentScopeToolSchema.prepareSubAgent(
                objectMapper, null, "child"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("must not be blank");
    }

    @Test
    void everyBuiltInSubAgentDeclaresAnExplicitStrictSchema() {
        for (AiAgentDefinition definition : new AiAgentRegistry().getAll()) {
            if (definition.getSubAgentTools() == null) {
                continue;
            }
            for (AiAgentDefinition.SubAgentToolDef child : definition.getSubAgentTools()) {
                AgentScopeToolSchema.PreparedSchema schema =
                        AgentScopeToolSchema.prepareSubAgent(
                                objectMapper,
                                child.getParametersSchema(),
                                child.getToolName());
                assertThat(schema.value()).containsEntry("additionalProperties", false);
            }
        }
    }
}
