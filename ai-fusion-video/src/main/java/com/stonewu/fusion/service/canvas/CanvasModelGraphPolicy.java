package com.stonewu.fusion.service.canvas;

import cn.hutool.json.JSONObject;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.capability.CanvasBatchDefinition;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasConfigValidator;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.generation.VideoModelRules;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * 编辑期验证新选择的提供商；完整输入能力和数量约束在运行计划中检查。
 */
@Component
@RequiredArgsConstructor
public class CanvasModelGraphPolicy implements CanvasGraphMutationPolicy {
    private static final Map<CanvasNodeKind, String> KINDS = Map.of(
            CanvasNodeKind.IMAGE_GENERATION, "image", CanvasNodeKind.VIDEO_GENERATION, "video",
            CanvasNodeKind.TEXT_GENERATION, "text", CanvasNodeKind.COMFYUI, "comfyui",
            CanvasNodeKind.MUSIC_GENERATION, "audio", CanvasNodeKind.SFX_GENERATION, "audio");
    private final CanvasModelCatalog models;
    private final CanvasCapabilityRegistry registry;

    @Override
    public void validate(CanvasDocument document, CanvasGraph before, CanvasGraphMutation mutation, CanvasActor actor) {
        for (var node : mutation.changes().upsertedNodes()) {
            var old = before.nodes().stream().filter(item -> item.id().equals(node.id())).findFirst().orElse(null);
            String kind = KINDS.get(node.kind());
            if (kind == null) {
                continue;
            }
            String field = kind.equals("audio") ? "apiConfigId" : "modelId";
            if (!node.config().has(field) || old != null && old.kind() == node.kind()
                    && Objects.equals(old.config().get(field), node.config().get(field))) {
                continue;
            }
            try {
                if (kind.equals("audio")) {
                    models.requireAudio(node.config().path(field).asLong());
                } else if (!kind.equals(models.require(node.config().path(field).asLong()).kind())) {
                    fail("模型类型与节点不匹配；ComfyUI 请使用工作流节点");
                }
            } catch (BusinessException invalid) {
                throw new CanvasException(CanvasErrorCode.INVALID_CONFIG, invalid.getMessage());
            }
        }
    }

    public void validateForRun(CanvasGraph graph, CanvasNode node, Map<String, Integer> observedOutputs) {
        validateNodes(graph, Set.of(node.id()), Set.of(), observedOutputs);
    }

    public void validateAll(CanvasGraph graph, Set<String> fixedEdges) {
        validateNodes(graph, new HashSet<>(graph.nodes().stream().map(CanvasNode::id).toList()), fixedEdges, Map.of());
    }

    /**
     * 导入模板保留未完成配置；仅核对已选择提供商的身份与工作流归属。
     */
    public void validateDraft(CanvasGraph graph) {
        for (var node : graph.nodes()) {
            var configured = node;
            if (node.kind() == CanvasNodeKind.FOREACH) {
                if (!node.config().hasNonNull("operationKind")
                        || !(node.config().get("operationConfig") instanceof ObjectNode body)) {
                    continue;
                }
                configured = new CanvasNode(node.id(),
                        CanvasNodeKind.fromCode(node.config().path("operationKind").asText()),
                        node.schemaVersion(), node.title(), node.layout(), body, null, 0, 0);
            }
            try {
                validateProviderSelection(configured);
            } catch (BusinessException invalid) {
                throw new CanvasException(CanvasErrorCode.INVALID_CONFIG, invalid.getMessage());
            }
        }
    }

    private void validateProviderSelection(CanvasNode node) {
        String kind = KINDS.get(node.kind());
        if (kind == null) {
            return;
        }
        if (kind.equals("audio")) {
            if (node.config().hasNonNull("apiConfigId")) {
                models.requireAudio(node.config().path("apiConfigId").asLong());
            }
            return;
        }
        if (!node.config().hasNonNull("modelId")) {
            return;
        }
        var model = models.require(node.config().path("modelId").asLong());
        if (!kind.equals(model.kind())) {
            fail("模型类型与节点不匹配；ComfyUI 请使用工作流节点");
        }
        if (!kind.equals("comfyui")) {
            return;
        }
        if (node.config().hasNonNull("workflowVersionId") && model.versions().stream()
                .noneMatch(version -> version.id() == node.config().path("workflowVersionId").asLong())) {
            fail("请选择此模型绑定工作流的已发布版本");
        }
        if (node.config().hasNonNull("resultType") && !Objects.equals(model.resultType(),
                node.config().path("resultType").asText())) {
            fail("工作流输出类型与模型不一致");
        }
    }

    private void validateNodes(CanvasGraph graph, Set<String> affected, Set<String> fixedEdges,
                               Map<String, Integer> observedOutputs) {
        for (var node : graph.nodes()) {
            if (!affected.contains(node.id())) {
                continue;
            }
            try {
                if (node.kind() == CanvasNodeKind.FOREACH && node.config().has("operationKind")) {
                    var body = CanvasBatchDefinition.body(node, registry, new CanvasConfigValidator());
                    validateConfigured(body, Map.of(node.config().path("inputPort").asText(), 1));
                } else if (KINDS.containsKey(node.kind())) {
                    validateNode(graph, node, fixedEdges, observedOutputs);
                }
            } catch (BusinessException invalid) {
                throw new CanvasException(CanvasErrorCode.INVALID_CONFIG, invalid.getMessage());
            }
        }
    }

    private void validateNode(CanvasGraph graph, CanvasNode node, Set<String> fixedEdges,
                              Map<String, Integer> observedOutputs) {
        Map<String, Integer> counts = new HashMap<>();
        for (var edge : graph.edges()) {
            if (edge.kind() != CanvasEdgeKind.EXECUTION || !edge.targetNodeId().equals(node.id())) {
                continue;
            }
            counts.merge(edge.targetPort(), edge.output() != null || fixedEdges.contains(edge.id()) ? 1
                    : knownCount(graph, edge.sourceNodeId(), edge.sourcePort(), new HashSet<>(), fixedEdges,
                    observedOutputs), Integer::sum);
        }
        validateConfigured(node, counts);
    }

    public void validateConfigured(CanvasNode node, Map<String, Integer> counts) {
        if (!KINDS.containsKey(node.kind())) {
            return;
        }
        if ("audio".equals(KINDS.get(node.kind()))) {
            if (!node.config().has("apiConfigId")) {
                fail("请选择音频服务");
            }
            models.requireAudio(node.config().path("apiConfigId").asLong());
            return;
        }
        if (!node.config().has("modelId")) {
            fail("请为“" + node.title() + "”选择模型");
        }
        var model = models.require(node.config().path("modelId").asLong());
        if (!Objects.equals(KINDS.get(node.kind()), model.kind())) {
            fail("模型类型与节点不匹配；ComfyUI 请使用工作流节点");
        }
        if ("comfyui".equals(model.kind())) {
            model = CanvasModelCatalog.forVersion(model, node.config().path("workflowVersionId").asLong());
            var version = model.versions().stream()
                    .filter(item -> item.id() == node.config().path("workflowVersionId").asLong())
                    .findFirst().orElseThrow(() -> new BusinessException("请选择此模型绑定工作流的已发布版本"));
            if (!model.resultType().equals(node.config().path("resultType").asText())) {
                fail("工作流输出类型与模型不一致");
            }
            var allowed = new HashSet<>(version.fields());
            allowed.addAll(List.of("modelId", "workflowVersionId", "resultType"));
            node.config().fieldNames().forEachRemaining(key -> {
                if (!allowed.contains(key)) {
                    fail("所选工作流未绑定参数：" + key);
                }
            });
        }
        var inputs = model.inputs();
        Set<String> connected = new HashSet<>();
        int images = 0;
        for (var port : registry.require(node.kind()).inputs()) {
            boolean literal = node.config().has(port.id());
            if (!counts.containsKey(port.id()) && !literal) {
                continue;
            }
            connected.add(port.id());
            var rule = inputs.get(port.id());
            if (rule == null) {
                fail("当前模型不支持" + port.label() + "，请更换模型或移除此输入");
            }
            int count = counts.getOrDefault(port.id(), literal ? 1 : 0);
            if (rule.maximum() != null && count > rule.maximum()) {
                fail(port.label() + "最多允许 " + rule.maximum() + " 项，当前至少 " + count + " 项");
            }
            if (Set.of("firstFrame", "lastFrame", "referenceImages").contains(port.id())) {
                images += count;
            }
        }
        if (model.parameters().get("maxImageInputs") instanceof Number max && images > max.intValue()) {
            fail("图片输入总量超过当前模型限制");
        }
        if (Boolean.TRUE.equals(model.parameters().get("exclusiveInputModes"))
                && (connected.contains("firstFrame") || connected.contains("lastFrame"))
                && connected.stream().anyMatch(port -> port.startsWith("reference"))) {
            fail("当前模型的首尾帧输入与多模态参考输入互斥");
        }
        var modelConfig = new JSONObject(model.parameters());
        var videoRules = VideoModelRules.from(modelConfig);
        if (videoRules != null) {
            var config = node.config();
            if (!connected.contains("prompt") && !connected.contains("context")) {
                boolean hasMedia = connected.stream().anyMatch(
                        port -> Set.of("firstFrame", "lastFrame", "referenceImages", "referenceVideos",
                                "referenceAudios").contains(port));
                videoRules.validatePrompt(config.path("prompt").asText(), hasMedia);
            }
            videoRules.resolve(modelConfig, new VideoModelRules.Parameters(
                            config.has("resolution") ? config.path("resolution").asText() : null,
                            config.has("ratio") ? config.path("ratio").asText() : null,
                            config.has("duration") ? config.path("duration").asInt() : null,
                            config.has("seed") ? config.path("seed").asLong() : null,
                            config.has("generateAudio") ? config.path("generateAudio").asBoolean() : null,
                            config.has("watermark") ? config.path("watermark").asBoolean() : null,
                            config.has("cameraFixed") ? config.path("cameraFixed").asBoolean() : null),
                    new VideoModelRules.Inputs(connected.contains("firstFrame"), connected.contains("lastFrame"),
                            connectedCount("referenceImages", counts, connected),
                            connectedCount("referenceVideos", counts, connected),
                            connectedCount("referenceAudios", counts, connected)));
            return;
        }
        choice(node, model, "ratio", "supportedAspectRatios");
        choice(node, model, "resolution", "supportedResolutions");
        if (node.config().has("duration")) {
            int duration = node.config().path("duration").asInt();
            if (duration <= 0) {
                fail("当前模型需要指定正数时长");
            }
            if (model.parameters().get("minDuration") instanceof Number min && duration < min.intValue()) {
                fail("时长低于模型允许范围");
            }
            if (model.parameters().get("maxDuration") instanceof Number max && duration > max.intValue()) {
                fail("时长超出模型允许范围");
            }
        }
        if (node.config().path("cameraFixed").asBoolean() && Boolean.FALSE.equals(
                model.parameters().get("supportCameraFixed"))) {
            fail("当前模型不支持固定镜头参数");
        }
    }

    private int connectedCount(String port, Map<String, Integer> counts, Set<String> connected) {
        return Math.max(counts.getOrDefault(port, 0), connected.contains(port) ? 1 : 0);
    }

    private void choice(CanvasNode node, CanvasModelCatalog.Option model, String field, String key) {
        Object values = model.parameters().get(key);
        if (node.config().has(field) && values instanceof List<?> list && !list.isEmpty()
                && !list.contains(node.config().path(field).asText())) {
            fail("当前模型不支持此" + field + "参数");
        }
    }

    private int knownCount(CanvasGraph graph, String id, String portId, Set<String> path, Set<String> fixedEdges,
                           Map<String, Integer> observedOutputs) {
        Integer observed = observedOutputs.get(id + ":" + portId);
        if (observed != null) {
            return observed;
        }
        if (!path.add(id)) {
            return 0;
        }
        var node = graph.nodes().stream().filter(item -> item.id().equals(id)).findFirst().orElseThrow();
        var port = registry.require(node.kind()).outputs().stream().filter(item -> item.id().equals(portId)).findFirst()
                .orElseThrow();
        if (!port.collection()) {
            return 1;
        }
        if (Set.of(CanvasNodeKind.IMAGE_GENERATION, CanvasNodeKind.VIDEO_GENERATION, CanvasNodeKind.COMFYUI)
                .contains(node.kind())) {
            return node.config().has("count") ? node.config().path("count").asInt() : 0;
        }
        if (node.kind() == CanvasNodeKind.COLLECTION) {
            return graph.edges().stream()
                    .filter(edge -> edge.kind() == CanvasEdgeKind.EXECUTION && edge.targetNodeId().equals(id))
                    .mapToInt(edge -> edge.output() != null || fixedEdges.contains(edge.id()) ? 1
                            : knownCount(graph, edge.sourceNodeId(), edge.sourcePort(), new HashSet<>(path), fixedEdges,
                            observedOutputs)).sum();
        }
        // 业务集合与未来运行产物的确切数量由执行计划校验，编辑期只拒绝已知超限。
        return 0;
    }

    private static void fail(String message) {
        throw new BusinessException(400, message);
    }
}
