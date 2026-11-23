package com.stonewu.fusion.service.canvas.run;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.collaboration.CanvasPinnedOutputs;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeActivity;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.State;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/** 读取连线指向的内容，文本组合与执行时共用规则。此入口不提交生成任务。 */
@Component
@RequiredArgsConstructor
public class CanvasGenerationInputPreview {
    /** 提供快捷生成面板的节点；三者共用提示词与内容设定的组合规则。 */
    private static final Set<CanvasNodeKind> QUICK_GENERATION_KINDS = Set.of(CanvasNodeKind.TEXT_GENERATION,
            CanvasNodeKind.IMAGE_GENERATION, CanvasNodeKind.VIDEO_GENERATION);

    private final CanvasDocumentRepository documents;
    private final CanvasGraphRepository graphs;
    private final CanvasAccessPolicy access;
    private final CanvasCapabilityRegistry capabilities;
    private final CanvasRunSources sources;
    private final CanvasPinnedOutputs fixed;
    private final CanvasRunService runs;
    private final CanvasValueExecutor values;

    @Transactional(propagation = Propagation.NEVER)
    public View read(String canvasId, String nodeId, long expectedRevision, CanvasActor actor) {
        CanvasDocument document = documents.require(canvasId, false);
        access.requireView(document, actor);
        requireRevision(document, expectedRevision);
        CanvasGraph graph = graphs.load(document);
        CanvasNode node = graph.nodes().stream().filter(item -> item.id().equals(nodeId)).findFirst()
                .orElseThrow(() -> CanvasRunException.invalid("节点已移除，请选择其他节点"));
        if (!QUICK_GENERATION_KINDS.contains(node.kind())) {
            throw CanvasRunException.invalid("请选择生成文本、生图或生视频节点");
        }
        Resolver resolver = new Resolver(document, graph, actor);
        List<Input> inputs = new ArrayList<>();
        List<Issue> issues = new ArrayList<>();
        Map<String, List<Value>> bound = new LinkedHashMap<>();
        for (CanvasEdge edge : resolver.incoming(nodeId)) {
            CanvasNode source = resolver.nodes.get(edge.sourceNodeId());
            try {
                List<Value> content = resolver.edge(edge);
                inputs.add(new Input(edge.id(), source.id(), source.title(), edge.targetPort(), content));
                bound.computeIfAbsent(edge.targetPort(), ignored -> new ArrayList<>()).addAll(content);
            } catch (BusinessException failure) {
                issues.add(new Issue(source.id(), source.title(), failure.getMessage()));
            }
        }
        String prompt = "";
        if (issues.isEmpty()) {
            // 自填内容由编辑区保留；这里只预览仍由连线提供的部分。
            var config = node.config().deepCopy();
            config.remove("additionalPrompt");
            CanvasNode connected = new CanvasNode(node.id(), node.kind(), node.schemaVersion(), node.title(),
                    node.layout(), config, node.resource(), node.configVersion(), node.layoutVersion());
            try {
                CanvasCreativeInputs.compose(connected, bound, sources);
                prompt = bound.getOrDefault("prompt", List.of()).stream().map(Value::text)
                        .collect(Collectors.joining("\n\n"));
            } catch (BusinessException failure) {
                issues.add(new Issue(node.id(), node.title(), failure.getMessage()));
            }
        }
        CanvasDocument current = documents.require(canvasId, false);
        access.requireView(current, actor);
        requireRevision(current, expectedRevision);
        return new View(current.revision(), prompt, List.copyOf(inputs), List.copyOf(issues));
    }

    private void requireRevision(CanvasDocument document, long expectedRevision) {
        if (document.revision() != expectedRevision) {
            throw CanvasRunException.conflict("连线已更新，请重新读取输入");
        }
    }

    private final class Resolver {
        private final CanvasDocument document;
        private final CanvasGraph graph;
        private final CanvasActor actor;
        private final Map<String, CanvasNode> nodes = new HashMap<>();
        private final Map<String, Ports> resolved = new HashMap<>();
        private Map<String, NodeActivity> activities;

        private Resolver(CanvasDocument document, CanvasGraph graph, CanvasActor actor) {
            this.document = document;
            this.graph = graph;
            this.actor = actor;
            graph.nodes().forEach(node -> nodes.put(node.id(), node));
        }

        private List<CanvasEdge> incoming(String id) {
            return graph.edges().stream()
                    .filter(edge -> edge.kind() == CanvasEdgeKind.EXECUTION && edge.targetNodeId().equals(id))
                    .sorted(Comparator.comparing(CanvasEdge::targetPort).thenComparing(CanvasEdge::inputIndex))
                    .toList();
        }

        private List<Value> edge(CanvasEdge edge) {
            List<Value> content = edge.output() == null
                    ? output(nodes.get(edge.sourceNodeId())).at(edge.sourcePort())
                    : List.of(fixed.resolve(document.id(), edge, actor));
            if (content.isEmpty()) {
                throw CanvasRunException.invalid("“" + nodes.get(edge.sourceNodeId()).title() + "”没有可用内容");
            }
            if (content.stream().anyMatch(value -> value.type() == CanvasValueType.TEXT
                    && (value.text() == null || value.text().isBlank()))) {
                throw CanvasRunException.invalid("“" + nodes.get(edge.sourceNodeId()).title() + "”的文字为空，请先补充内容");
            }
            content.forEach(value -> sources.authorize(value, actor, true));
            return content;
        }

        private Ports output(CanvasNode node) {
            if (resolved.containsKey(node.id())) {
                return resolved.get(node.id());
            }
            Ports result;
            String executor = capabilities.require(node.kind()).executor();
            if (executor.equals("constant") || executor.equals("resource")) {
                Set<String> ports = graph.edges().stream()
                        .filter(edge -> edge.kind() == CanvasEdgeKind.EXECUTION && edge.sourceNodeId().equals(node.id()))
                        .map(CanvasEdge::sourcePort).collect(Collectors.toSet());
                result = sources.freeze(document, node, actor, ports);
            } else if (node.kind() == CanvasNodeKind.TEXT_COMBINE || node.kind() == CanvasNodeKind.COLLECTION) {
                Map<String, List<Value>> inputs = new LinkedHashMap<>();
                for (CanvasEdge edge : incoming(node.id())) {
                    inputs.computeIfAbsent(edge.targetPort(), ignored -> new ArrayList<>()).addAll(edge(edge));
                }
                result = values.resolve(node, new Ports(inputs), Ports.empty()).outputs();
            } else {
                if (activities == null) {
                    activities = new HashMap<>();
                    runs.nodeActivities(document.id(), actor).forEach(item -> activities.put(item.node().nodeId(), item));
                }
                NodeActivity activity = activities.get(node.id());
                if (activity == null || activity.node().status() != State.SUCCEEDED
                        || !"CURRENT".equals(activity.node().derivedState()) || activity.node().contentUnavailable()) {
                    throw CanvasRunException.invalid("请先完成“" + node.title() + "”并选用结果");
                }
                result = activity.node().outputs();
            }
            resolved.put(node.id(), result);
            return result;
        }
    }

    public record Input(String edgeId, String nodeId, String title, String port, List<Value> values) {
    }

    public record Issue(String nodeId, String title, String message) {
    }

    public record View(long documentRevision, String prompt, List<Input> inputs, List<Issue> issues) {
    }
}
