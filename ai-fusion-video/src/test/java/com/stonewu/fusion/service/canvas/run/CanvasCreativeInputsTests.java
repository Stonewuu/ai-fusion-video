package com.stonewu.fusion.service.canvas.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CanvasCreativeInputsTests {
    private final ObjectMapper json = new ObjectMapper();
    private final CanvasRunSources sources = mock(CanvasRunSources.class);
    private final CanvasNode node = mock(CanvasNode.class);

    @ParameterizedTest
    @EnumSource(value = CanvasNodeKind.class, names = {"IMAGE_GENERATION", "VIDEO_GENERATION", "TEXT_GENERATION", "COMFYUI"})
    void generationAcceptsDifferentContentSourcesButKeepsMediaInputsExplicit(CanvasNodeKind kind) {
        var capability = new CanvasCapabilityRegistry().require(kind);
        var context = capability.inputs().stream().filter(port -> port.id().equals("context")).findFirst().orElseThrow();
        for (var type : List.of(CanvasValueType.TEXT, CanvasValueType.ASSET, CanvasValueType.ASSET_ITEM, CanvasValueType.SHOT, CanvasValueType.SCRIPT_SCENE))
            assertThat(context.accepts(type, json.createObjectNode())).isTrue();
        assertThat(context.accepts(CanvasValueType.IMAGE, json.createObjectNode())).isFalse();
        assertThat(context.accepts(CanvasValueType.VIDEO, json.createObjectNode())).isFalse();
    }

    @Test
    void assetsCanBecomeVideoInstructionsWithoutCreatingStoryboardOrImplicitImageInputs() {
        var data = json.createObjectNode().put("name", "阿青").put("description", "穿蓝色雨衣，站在灯塔门口")
                .put("imageUrl", "media://private-image").put("projectId", 7);
        var asset = new Value("asset", CanvasValueType.ASSET_ITEM, null, null, null,
                new CanvasResourceReference(CanvasValueType.ASSET_ITEM, 8L, 2L), data, null);
        var inputs = prepare(Map.of("context", List.of(asset), "prompt", List.of(text("让角色抬头看向灯塔"))));
        CanvasCreativeInputs.compose(node, inputs, sources);
        assertThat(inputs.get("prompt").getFirst().text()).contains("让角色抬头看向灯塔", "阿青", "穿蓝色雨衣")
                .doesNotContain("media://", "projectId");
        assertThat(inputs).doesNotContainKeys("firstFrame", "referenceImages");
        assertThat(data.path("imageUrl").asText()).isEqualTo("media://private-image");
        assertThat(inputs.get("context")).containsExactly(asset);
    }

    @Test
    void multipleContextsPreserveOrderAndAnExplicitImageRemainsAnImage() {
        var image = new Value("frame", CanvasValueType.IMAGE, null, "frame-id", 1, null, null, null);
        var inputs = prepare(Map.of("context", List.of(text("角色设定"), text("场景设定")), "firstFrame", List.of(image)));
        CanvasCreativeInputs.compose(node, inputs, sources);
        assertThat(inputs.get("prompt").getFirst().text()).containsSubsequence("角色设定", "场景设定");
        assertThat(inputs.get("firstFrame")).containsExactly(image);
    }

    @Test
    void oversizedAndEmptyContextFailInsteadOfSilentlyDroppingUserMaterial() {
        var large = prepare(Map.of("context", List.of(text("文".repeat(32001)))));
        assertThatThrownBy(() -> CanvasCreativeInputs.compose(node, large, sources)).hasMessageContaining("超过");
        var empty = prepare(Map.of("context", List.of(text(" "))));
        assertThatThrownBy(() -> CanvasCreativeInputs.compose(node, empty, sources)).hasMessageContaining("没有可用文本");
    }

    private Map<String, List<Value>> prepare(Map<String, List<Value>> values) {
        when(node.kind()).thenReturn(CanvasNodeKind.VIDEO_GENERATION);
        when(sources.text(anyString())).thenAnswer(call -> text(call.getArgument(0)));
        return new LinkedHashMap<>(values);
    }

    private Value text(String content) {
        return new Value("text", CanvasValueType.TEXT, content, null, null, null, null, null);
    }
}
