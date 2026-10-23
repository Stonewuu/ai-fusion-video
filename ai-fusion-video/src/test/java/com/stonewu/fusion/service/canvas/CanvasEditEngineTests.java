package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.exc.MismatchedInputException;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasConfigValidator;
import com.stonewu.fusion.service.canvas.command.CanvasEditEngine;
import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.CreateNode;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.DeleteNode;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.LayoutNode;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.PatchNode;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class CanvasEditEngineTests {
    private final ObjectMapper json = new ObjectMapper();
    private final CanvasEditEngine engine = new CanvasEditEngine(new CanvasGraphValidator(
            new CanvasCapabilityRegistry(), new CanvasConfigValidator()));

    @Test
    void checksConfigurationAndLayoutVersionsIndependently() {
        CanvasNode original = text();
        CanvasGraphMutation moved = engine.apply(graph(original), List.of(new LayoutNode(original.id(), 0, layout(100, 200, null))));
        CanvasGraphMutation edited = engine.apply(moved.graph(), List.of(new PatchNode(original.id(), 0, null,
                json.createObjectNode().put("content", "新的内容"))));
        CanvasNode result = edited.graph().nodes().getFirst();
        assertThat(result.layout().x()).isEqualTo(100);
        assertThat(result.configVersion()).isEqualTo(1);
        assertThat(result.layoutVersion()).isEqualTo(1);
        assertThat(result.config().path("content").asText()).isEqualTo("新的内容");
    }

    @Test
    void rejectsStaleWritesAndDoesNotApplyEarlierOperationsInTheBatch() {
        CanvasNode original = text();
        CanvasGraph graph = graph(original);
        List<CanvasGraphOperation> changes = List.of(new PatchNode(original.id(), 0, null,
                        json.createObjectNode().put("content", "批次第一步")),
                new PatchNode(original.id(), 0, "过期修改", null));
        assertThatThrownBy(() -> engine.apply(graph, changes)).isInstanceOfSatisfying(CanvasException.class,
                error -> assertThat(error.getErrorCode()).isEqualTo(CanvasErrorCode.VERSION_CONFLICT));
        assertThat(graph.nodes().getFirst().config().path("content").asText()).isEqualTo("原内容");
        assertThat(graph.nodes().getFirst().configVersion()).isZero();
    }

    @Test
    void rejectsAnInvalidFinalGraphWithoutMutatingTheOriginal() {
        CanvasNode original = text();
        assertThatThrownBy(() -> engine.apply(graph(original), List.of(new PatchNode(original.id(), 0, null,
                json.createObjectNode().put("unsupported", "不能存入"))))).isInstanceOf(CanvasException.class);
        assertThat(original.config().has("unsupported")).isFalse();
    }

    @Test
    void detachesGroupChildrenWithoutChangingTheirWorldPosition() {
        String groupId = UUID.randomUUID().toString();
        CanvasNode group = new CanvasNode(groupId, CanvasNodeKind.GROUP, 1, "场次", layout(100, 200, null),
                json.createObjectNode(), null, 0, 0);
        CanvasNode child = new CanvasNode(UUID.randomUUID().toString(), CanvasNodeKind.TEXT, 1, "对白",
                layout(20, 30, groupId), json.createObjectNode(), null, 0, 0);
        CanvasGraphMutation mutation = engine.apply(new CanvasGraph(1, null, List.of(group, child), List.of()),
                List.of(new DeleteNode(groupId, 0, 0)));
        CanvasNode detached = mutation.graph().nodes().getFirst();
        assertThat(detached.layout().parentId()).isNull();
        assertThat(detached.layout().x()).isEqualTo(120);
        assertThat(detached.layout().y()).isEqualTo(230);
        assertThat(detached.layoutVersion()).isEqualTo(1);
        assertThat(mutation.changes().deletedNodeIds()).containsExactly(groupId);
        assertThat(mutation.previousValues().upsertedNodes()).contains(group, child);
    }

    @Test
    void requiresExplicitUnlockBeforeMovingLockedNodes() {
        CanvasNode original = text();
        CanvasNode locked = new CanvasNode(original.id(), original.kind(), 1, original.title(),
                new CanvasNodeLayout(0, 0, null, null, null, true, false), original.config(), null, 0, 0);
        assertThatThrownBy(() -> engine.apply(graph(locked), List.of(new LayoutNode(locked.id(), 0, layout(10, 10, null)))))
                .isInstanceOf(CanvasException.class).hasMessageContaining("锁定");
        CanvasGraphMutation result = engine.apply(graph(locked), List.of(
                new LayoutNode(locked.id(), 0, layout(0, 0, null)),
                new LayoutNode(locked.id(), 1, layout(10, 10, null))));
        assertThat(result.graph().nodes().getFirst().layoutVersion()).isEqualTo(2);
    }

    @Test
    void recordsOnlyAffectedNodesAndPreservesValuesForUndo() {
        CanvasNode first = text();
        CanvasNode second = text();
        CanvasGraphMutation mutation = engine.apply(new CanvasGraph(1, null, List.of(first, second), List.of()),
                List.of(new PatchNode(first.id(), 0, "改名", null)));
        assertThat(mutation.changes().upsertedNodes()).extracting(CanvasNode::id).containsExactly(first.id());
        assertThat(mutation.previousValues().upsertedNodes()).containsExactly(first);
        assertThat(mutation.graph().nodes().get(1)).isEqualTo(second);
    }

    @Test
    void supportsExplicitConfigurationRemoval() {
        CanvasNode original = text();
        ObjectNode patch = json.createObjectNode().putNull("content");
        CanvasGraphMutation mutation = engine.apply(graph(original), List.of(new PatchNode(original.id(), 0, null, patch)));
        assertThat(mutation.graph().nodes().getFirst().config().has("content")).isFalse();
    }

    @Test
    void rejectsMissingExpectedVersionInsteadOfAssumingZero() {
        String request = """
                {"type":"canvas.node.patch","nodeId":"id","title":"修改"}
                """;
        assertThatThrownBy(() -> json.readValue(request, CanvasGraphOperation.class))
                .isInstanceOf(MismatchedInputException.class).hasMessageContaining("expectedConfigVersion");
    }

    @Test
    void copiesIncomingConfigurationSoCallerCannotChangeCommittedMutation() {
        CanvasNode incoming = text();
        CanvasGraphMutation mutation = engine.apply(new CanvasGraph(1, null, List.of(), List.of()),
                List.of(new CreateNode(incoming)));
        incoming.config().put("content", "调用者随后改了输入对象");
        assertThat(mutation.graph().nodes().getFirst().config().path("content").asText()).isEqualTo("原内容");
    }

    private CanvasNode text() {
        return new CanvasNode(UUID.randomUUID().toString(), CanvasNodeKind.TEXT, 1, "文字", layout(0, 0, null),
                json.createObjectNode().put("content", "原内容"), null, 0, 0);
    }

    private CanvasNodeLayout layout(double x, double y, String parent) {
        return new CanvasNodeLayout(x, y, null, null, parent, false, false);
    }

    private CanvasGraph graph(CanvasNode node) { return new CanvasGraph(1, null, List.of(node), List.of()); }
}
