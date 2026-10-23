package com.stonewu.fusion.service.canvas.template;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

import static com.stonewu.fusion.service.canvas.template.CanvasTemplateData.ProviderKind;
import static com.stonewu.fusion.service.canvas.template.CanvasTemplateData.Use;

/**
 * 明确识别节点和批量子操作的提供商字段，不把 MCP 参数正文里的 modelId 当成本平台模型编号。
 */
@Component
@RequiredArgsConstructor
public class CanvasTemplateProviders {
    private static final List<String> FIELDS = List.of("modelId", "imageModelId", "videoModelId", "apiConfigId",
            "workflowVersionId", "contractId");
    private final CanvasCapabilityRegistry registry;

    public record Found(Use use, ProviderKind kind, JsonNode id) {
    }

    public List<Found> references(CanvasNode node) {
        List<Found> result = new ArrayList<>();
        collect(node, false, result);
        if (node.kind() == CanvasNodeKind.FOREACH && node.config().path("operationConfig").isObject()) {
            if (node.config().hasNonNull("operationKind")) {
                collect(node, true, result);
            } else if (!node.config().path("operationConfig").isEmpty()) {
                throw invalid("请先选择批量操作，或清空未使用的批量配置");
            }
        }
        return List.copyOf(result);
    }

    private void collect(CanvasNode node, boolean batch, List<Found> result) {
        ObjectNode config = config(node, batch);
        for (String field : FIELDS) {
            if (config.hasNonNull(field)) {
                var use = new Use(node.id(), batch, field);
                result.add(new Found(use, kind(node, use), config.get(field)));
            }
        }
    }

    public ProviderKind kind(CanvasNode node, Use use) {
        if (!node.id().equals(use.nodeId())) {
            throw invalid("提供商绑定不属于此节点");
        }
        CanvasNodeKind kind = operation(node, use.batch());
        if (!FIELDS.contains(use.field()) || !registry.require(kind).configFields().containsKey(use.field())) {
            throw invalid("节点未声明此提供商字段");
        }
        return switch (use.field()) {
            case "imageModelId" -> ProviderKind.IMAGE_MODEL;
            case "videoModelId" -> ProviderKind.VIDEO_MODEL;
            case "apiConfigId" -> ProviderKind.AUDIO_API;
            case "workflowVersionId" -> ProviderKind.WORKFLOW;
            case "contractId" -> ProviderKind.MCP;
            case "modelId" -> switch (kind) {
                case IMAGE_GENERATION -> ProviderKind.IMAGE_MODEL;
                case VIDEO_GENERATION -> ProviderKind.VIDEO_MODEL;
                case COMFYUI -> ProviderKind.COMFYUI_MODEL;
                default -> ProviderKind.TEXT_MODEL;
            };
            default -> throw invalid("未知提供商字段");
        };
    }

    public ObjectNode config(CanvasNode node, boolean batch) {
        if (!batch) {
            return node.config();
        }
        operation(node, true);
        if (!(node.config().get("operationConfig") instanceof ObjectNode nested)) {
            throw invalid("批量操作配置不完整");
        }
        return nested;
    }

    private CanvasNodeKind operation(CanvasNode node, boolean batch) {
        if (!batch) {
            return node.kind();
        }
        if (node.kind() != CanvasNodeKind.FOREACH) {
            throw invalid("只有批量节点可以绑定子操作");
        }
        CanvasNodeKind kind = CanvasNodeKind.fromCode(node.config().path("operationKind").asText());
        if (kind == CanvasNodeKind.FOREACH) {
            throw invalid("模板不接受嵌套批量操作");
        }
        return kind;
    }

    private BusinessException invalid(String text) {
        return new BusinessException(400, text);
    }
}
