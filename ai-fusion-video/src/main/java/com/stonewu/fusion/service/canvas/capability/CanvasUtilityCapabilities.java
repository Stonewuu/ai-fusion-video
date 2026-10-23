package com.stonewu.fusion.service.canvas.capability;

import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeTypes;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.capability.CanvasConfigField.*;
import static com.stonewu.fusion.service.canvas.capability.CanvasDefinitions.*;

final class CanvasUtilityCapabilities {
    private CanvasUtilityCapabilities() {
    }

    static List<CanvasNodeCapability> definitions() {
        String[] types = Arrays.stream(CanvasValueType.values()).map(CanvasValueType::code).toArray(String[]::new);
        String[] agentOutputs = Arrays.stream(CanvasValueType.values())
                .filter(type -> type != CanvasValueType.PROJECT && type != CanvasValueType.AUDIO && type != CanvasValueType.DOCUMENT)
                .map(CanvasValueType::code).toArray(String[]::new);
        return List.of(
                textNode(CanvasNodeKind.TEXT, "文本"), textNode(CanvasNodeKind.STORY, "故事"),
                capability(CanvasNodeKind.NOTE, "备注", "organization", "none", false,
                        fields(field("content", text(64000))), List.of(), List.of()),
                capability(CanvasNodeKind.GROUP, "分组", "organization", "none", false, Map.of(), List.of(), List.of()),
                capability(CanvasNodeKind.TEXT_COMBINE, "拼接文本", "processing", "text", false,
                        fields(field("separator", text(256))), List.of(port("texts", CanvasValueType.TEXT, true, true)),
                        List.of(port("text", CanvasValueType.TEXT, false, false))),
                capability(CanvasNodeKind.TEXT_GENERATION, "生成文本", "generation", "agent", false,
                        fields(field("modelId", integer(1, MAX_SAFE_ID, false)), field("prompt", text(64000)),
                                field("instruction", text(32000))),
                        List.of(port("prompt", CanvasValueType.TEXT, false, true), contextPort(),
                                port("images", CanvasValueType.IMAGE, true, false)),
                        List.of(port("text", CanvasValueType.TEXT, false, false))),
                capability(CanvasNodeKind.AGENT, "智能体", "creation", "agent", false,
                        fields(field("agentType",
                                        choice(false, CanvasAgentNodeTypes.TYPES.stream().sorted().toArray(String[]::new))),
                                field("modelId", integer(1, MAX_SAFE_ID, false)),
                                field("imageModelId", integer(1, MAX_SAFE_ID, false)),
                                field("videoModelId", integer(1, MAX_SAFE_ID, false)),
                                field("maxImages", integer(0, 1000, false)),
                                field("maxVideoSeconds", integer(0, 36000, false)),
                                field("maxAgentRuns", integer(1, 64, false)),
                                field("frameType", choice(false, "first", "last")),
                                field("skillMode", choice(false, "INHERIT", "CUSTOM")),
                                field("skillNames", stringList(8, 100)),
                                field("instruction", text(64000)), field("inputType", choice(true, types)),
                                field("resultType", choice(true, agentOutputs))),
                        agentInputs(port("instruction", CanvasValueType.TEXT, false, false),
                                typedPort("inputs", "inputType", true, false)),
                        List.of(typedPort("results", "resultType", true, false),
                                port("summary", CanvasValueType.TEXT, false, false))),
                capability(CanvasNodeKind.COLLECTION, "集合", "control", "control", false,
                        fields(field("valueType", choice(true, types))),
                        List.of(typedPort("items", "valueType", true, false)),
                        List.of(typedPort("items", "valueType", true, false))),
                capability(CanvasNodeKind.MCP_TOOL, "MCP 工具", "processing", "mcp", false,
                        fields(field("contractId", text(36)), field("contractVersion", integer(0, MAX_SAFE_ID, true)),
                                field("argumentsJson", text(128000)), field("input", text(128000)),
                                field("inputMode", choice(true, "NONE", "TEXT_FIELD", "JSON_OBJECT")),
                                field("inputParameter", text(256))),
                        List.of(port("input", CanvasValueType.TEXT, false, false)),
                        List.of(port("text", CanvasValueType.TEXT, false, false))),
                capability(CanvasNodeKind.SELECTION, "选择候选", "control", "control", false,
                        fields(field("valueType", choice(true, types)), field("selectedOutputId", text(128))),
                        List.of(typedPort("candidates", "valueType", true, true)),
                        List.of(typedPort("selected", "valueType", false, false))),
                capability(CanvasNodeKind.FOREACH, "批量处理", "control", "control", false,
                        fields(field("inputType", choice(true, types)), field("resultType", choice(true, types)),
                                field("maxItems", integer(1, 1000, true)),
                                field("operationKind", choice(false,
                                        CanvasBatchDefinition.OPERATIONS.stream().map(CanvasNodeKind::code)
                                                .toArray(String[]::new))),
                                field("operationConfig", object()), field("inputPort", text(64)),
                                field("outputPort", text(64))),
                        List.of(typedPort("items", "inputType", true, true)),
                        List.of(typedPort("results", "resultType", true, false))),
                capability(CanvasNodeKind.WRITEBACK, "应用到项目", "delivery", "writeback", false,
                        fields(field("binding", choice(true,
                                        CanvasWritebackFields.list().stream().map(CanvasWritebackFields.Field::key)
                                                .toArray(String[]::new))),
                                field("targetType", choice(true, Arrays.stream(CanvasValueType.values())
                                        .filter(CanvasValueType::isBusinessResource).map(CanvasValueType::code)
                                        .toArray(String[]::new))),
                                field("valueType", choice(true, "text", "image", "video", "audio"))),
                        List.of(new CanvasPortDefinition("value", "写入内容", null, "valueType", false, true, 1),
                                new CanvasPortDefinition("target", "目标对象", null, "targetType", false, true, 1)),
                        List.of(new CanvasPortDefinition("resource", "更新后的对象", null, "targetType", false, false,
                                1))),
                capability(CanvasNodeKind.EXPORT, "导出", "delivery", "export", false,
                        fields(field("valueType", choice(true, types)),
                                field("format", choice(true, "media", "markdown", "csv", "json", "zip"))),
                        List.of(typedPort("items", "valueType", true, true)), List.of())
        );
    }

    private static CanvasNodeCapability textNode(CanvasNodeKind kind, String label) {
        return capability(kind, label, "content", "constant", false,
                fields(field("content", text(128000))), List.of(),
                List.of(port("text", CanvasValueType.TEXT, false, false)));
    }
}
