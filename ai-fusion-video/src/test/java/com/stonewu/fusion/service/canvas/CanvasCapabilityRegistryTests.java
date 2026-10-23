package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasConfigField;
import com.stonewu.fusion.service.canvas.capability.CanvasConfigValidator;
import com.stonewu.fusion.service.canvas.capability.CanvasNodeCapability;
import com.stonewu.fusion.service.canvas.capability.CanvasPortDefinition;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class CanvasCapabilityRegistryTests {
    private final CanvasCapabilityRegistry registry = new CanvasCapabilityRegistry();

    @Test
    void coversAllDeclaredNodeKindsWithUnambiguousPorts() {
        assertThat(registry.list()).extracting(CanvasNodeCapability::kind).containsExactly(CanvasNodeKind.values());
        for (CanvasNodeCapability definition : registry.list()) {
            assertThat(definition.inputs()).extracting(CanvasPortDefinition::id).doesNotHaveDuplicates();
            assertThat(definition.outputs()).extracting(CanvasPortDefinition::id).doesNotHaveDuplicates();
            assertDynamicFields(definition, definition.inputs());
            assertDynamicFields(definition, definition.outputs());
        }
    }

    @Test
    void serializesStablePublicNodeCodesAndCapabilityMetadata() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        String encoded = mapper.writeValueAsString(registry.require(CanvasNodeKind.VIDEO_GENERATION));
        assertThat(mapper.readTree(encoded).path("kind").asText()).isEqualTo("videoGeneration");
        assertThat(mapper.readTree(encoded).path("configFields").path("count").path("minimum").asDouble()).isEqualTo(1);
        assertThat(mapper.readValue("\"scriptScene\"", CanvasValueType.class)).isEqualTo(CanvasValueType.SCRIPT_SCENE);
        assertThat(mapper.readValue("\"videoGeneration\"", CanvasNodeKind.class)).isEqualTo(CanvasNodeKind.VIDEO_GENERATION);
    }

    @Test
    void keepsAssetExtractionSeparateFromDestructiveFullScriptParsing() {
        assertThat(registry.require(CanvasNodeKind.ASSET_EXTRACT).configFields().get("agentType").enumValues())
                .containsExactly("canvas_asset_extractor");
    }

    @Test
    void globalAssetReferencesCanBeUsedWithoutWeakeningProjectResources() {
        assertThat(registry.require(CanvasNodeKind.ASSET).requiresProject()).isFalse();
        assertThat(registry.require(CanvasNodeKind.ASSET_ITEM).requiresProject()).isFalse();
        assertThat(registry.require(CanvasNodeKind.SCRIPT).requiresProject()).isTrue();
        assertThat(registry.require(CanvasNodeKind.SHOT).requiresProject()).isTrue();
    }

    @Test
    void supportsDifferentInputAndResultTypesForAgentAndBatchOperations() {
        var config = new ObjectMapper().createObjectNode().put("inputType", "shot")
                .put("resultType", "video").put("maxItems", 50);
        CanvasNodeCapability definition = registry.require(CanvasNodeKind.FOREACH);
        new CanvasConfigValidator().validate(definition, config);
        assertThat(definition.inputs().getFirst().resolveType(config)).isEqualTo(CanvasValueType.SHOT);
        assertThat(definition.outputs().getFirst().resolveType(config)).isEqualTo(CanvasValueType.VIDEO);
    }

    private void assertDynamicFields(CanvasNodeCapability definition, List<CanvasPortDefinition> ports) {
        for (CanvasPortDefinition port : ports) {
            if (port.valueTypeField() == null) continue;
            CanvasConfigField field = definition.configFields().get(port.valueTypeField());
            assertThat(field).isNotNull();
            assertThat(field.required()).isTrue();
            assertThat(field.enumValues()).isNotEmpty();
            field.enumValues().forEach(CanvasValueType::fromCode);
        }
    }
}
