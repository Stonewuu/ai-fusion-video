package com.stonewu.fusion.service.canvas.capability;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.*;

import java.util.List;

/**
 * 批量体使用现有节点契约，禁止任意执行器名称、嵌套循环或不受校验的参数。
 */
public final class CanvasBatchDefinition {
    private CanvasBatchDefinition() {
    }

    public static final List<CanvasNodeKind> OPERATIONS = List.of(CanvasNodeKind.TEXT_GENERATION,
            CanvasNodeKind.IMAGE_GENERATION, CanvasNodeKind.VIDEO_GENERATION, CanvasNodeKind.COMFYUI,
            CanvasNodeKind.IMAGE_TRANSFORM, CanvasNodeKind.FRAME_EXTRACT, CanvasNodeKind.MUSIC_GENERATION,
            CanvasNodeKind.SFX_GENERATION, CanvasNodeKind.VIDEO_COMPOSE, CanvasNodeKind.AUDIO_MIX,
            CanvasNodeKind.TEXT_COMBINE, CanvasNodeKind.MCP_TOOL, CanvasNodeKind.AGENT, CanvasNodeKind.SCRIPT_PARSE,
            CanvasNodeKind.STORY_TO_SCRIPT, CanvasNodeKind.EPISODE_PARSE, CanvasNodeKind.STORYBOARD_GENERATE,
            CanvasNodeKind.ASSET_EXTRACT, CanvasNodeKind.ASSET_IMAGE_GENERATE, CanvasNodeKind.FRAME_GENERATE);

    public static CanvasNode body(CanvasNode parent, CanvasCapabilityRegistry registry,
                                  CanvasConfigValidator validator) {
        var config = parent.config();
        if (!config.hasNonNull("operationKind") || !config.hasNonNull("operationConfig")
                || !config.hasNonNull("inputPort") || !config.hasNonNull("outputPort")) {
            fail("请配置批量处理的操作、输入和输出");
        }
        var kind = CanvasNodeKind.fromCode(config.path("operationKind").asText());
        if (!OPERATIONS.contains(kind)) {
            fail("批量处理不支持此操作，也不能嵌套批量处理");
        }
        if (!(config.get("operationConfig") instanceof ObjectNode)) {
            fail("批量操作配置必须为对象");
        }
        var operation = (ObjectNode) config.get("operationConfig");
        var capability = registry.require(kind);
        validator.validate(capability, operation);
        var input = capability.inputs().stream().filter(port -> port.id().equals(config.path("inputPort").asText()))
                .findFirst()
                .orElseThrow(() -> invalid("批量操作的输入端口不存在"));
        var output = capability.outputs().stream().filter(port -> port.id().equals(config.path("outputPort").asText()))
                .findFirst()
                .orElseThrow(() -> invalid("批量操作的输出端口不存在"));
        if (!input.accepts(CanvasValueType.fromCode(config.path("inputType").asText()), operation)
                || !output.resolveType(operation).code().equals(config.path("resultType").asText())) {
            fail("批量输入或结果类型与所选操作不匹配");
        }
        if (operation.has(input.id())) {
            fail("逐项输入不能同时配置固定值");
        }
        for (var port : capability.inputs()) {
            if (port.id().equals(input.id())) {
                continue;
            }
            if (port.required() && !(port.id().equals("prompt") && input.id().equals("context"))
                    && (!operation.hasNonNull(port.id()) || operation.path(port.id()).asText().isBlank())) {
                fail("请填写批量操作的“" + (port.id().equals("prompt") ? "提示词" : port.label()) + "”");
            }
            if (operation.has(port.id()) && !port.accepts(CanvasValueType.TEXT, operation)) {
                fail("批量操作不能用文字代替媒体输入");
            }
        }
        if (kind == CanvasNodeKind.FRAME_EXTRACT && !operation.path("frameType").asText().equals("both")
                && !output.id().equals(operation.path("frameType").asText() + "Frame")) {
            fail("请选择实际提取的帧作为输出");
        }
        int count = operation.path("count").asInt(1);
        if (count < 1 || count > 32 || (long) config.path("maxItems").asInt() * count > 1000) {
            fail("批量结果总数最多为 1000，请减少处理数量或每项生成数量");
        }
        return new CanvasNode(parent.id(), kind, capability.schemaVersion(), parent.title(), parent.layout(),
                operation.deepCopy(), null, parent.configVersion(), parent.layoutVersion());
    }

    public static void draft(CanvasNode node, CanvasCapabilityRegistry registry, CanvasConfigValidator validator) {
        if (node.kind() != CanvasNodeKind.FOREACH || !node.config().has("operationKind")) {
            return;
        }
        var config = node.config();
        var kind = CanvasNodeKind.fromCode(config.path("operationKind").asText());
        if (!OPERATIONS.contains(kind)) {
            fail("批量处理不支持此操作，也不能嵌套批量处理");
        }
        var capability = registry.require(kind);
        ObjectNode operation = config.has("operationConfig")
                ? (ObjectNode) config.get("operationConfig") : config.objectNode();
        validator.validateDraft(capability, operation);
        draftPort(config, operation, capability.inputs(), "inputPort", "inputType");
        draftPort(config, operation, capability.outputs(), "outputPort", "resultType");
        if (config.has("inputPort") && operation.has(config.path("inputPort").asText())) {
            fail("逐项输入不能同时配置固定值");
        }
    }

    private static void draftPort(ObjectNode config, ObjectNode operation, List<CanvasPortDefinition> ports,
                                  String portField, String typeField) {
        if (!config.has(portField)) {
            return;
        }
        var port = ports.stream().filter(item -> item.id().equals(config.path(portField).asText())).findFirst()
                .orElseThrow(() -> invalid("批量操作的输入或输出端口不存在"));
        if (!config.has(typeField) || port.valueTypeField() != null && !operation.has(port.valueTypeField())) {
            return;
        }
        if (!port.accepts(CanvasValueType.fromCode(config.path(typeField).asText()), operation)) {
            fail("批量输入或结果类型与所选操作不匹配");
        }
    }

    private static void fail(String message) {
        throw invalid(message);
    }

    private static CanvasException invalid(String message) {
        return new CanvasException(CanvasErrorCode.INVALID_CONFIG, message);
    }
}
