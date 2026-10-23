package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Applies explicit platform bindings to a deep copy of an immutable workflow version.
 */
@Component
@RequiredArgsConstructor
public class ComfyUiWorkflowRenderer {

    private final ObjectMapper objectMapper;
    private final ComfyUiWorkflowDocumentService documentService;

    public ObjectNode render(int modelType,
                             ComfyUiWorkflowVersion version,
                             Map<String, Object> values) {
        if (version == null) {
            throw new BusinessException(400, "ComfyUI 工作流版本不能为空");
        }
        ObjectNode workflow = documentService.parseApiWorkflow(version.getApiWorkflowJson()).deepCopy();
        List<ComfyUiInputBinding> bindings = documentService.parseInputBindings(
                modelType, version.getApiWorkflowJson(), version.getInputBindingsJson());
        Map<String, JsonNode> writtenInputs = new HashMap<>();
        for (ComfyUiInputBinding binding : bindings) {
            if (values == null || !values.containsKey(binding.businessField())) {
                continue;
            }
            Object rawValue = values.get(binding.businessField());
            if (rawValue == null) {
                continue;
            }
            Object selectedValue = selectIndexedValue(rawValue, binding);
            JsonNode renderedValue = convertValue(selectedValue, binding);
            String targetNodeId = binding.nodeId();
            String targetInputName = binding.inputName();
            ObjectNode nodeInputs = (ObjectNode) workflow.path(targetNodeId).path("inputs");
            JsonNode current = nodeInputs.get(binding.inputName());
            if (isUploadedMedia(binding.valueType()) && isLink(current)) {
                JsonNode sourceNode = workflow.path(current.get(0).asText());
                // 只支持内置 LoadImage 的 IMAGE 输出；其他链路必须明确绑定加载节点的文件字段。
                if (!"uploaded_image".equals(binding.valueType())
                        || current.get(1).asInt() != 0
                        || !"LoadImage".equals(sourceNode.path("class_type").asText())
                        || !sourceNode.path("inputs").path("image").isTextual()) {
                    throw new BusinessException(400,
                            "上传媒体请直接绑定加载节点的文件输入；图片连线仅支持 LoadImage 的 IMAGE 输出: "
                                    + binding.nodeId() + "." + binding.inputName());
                }
                targetNodeId = current.get(0).asText();
                targetInputName = "image";
                nodeInputs = (ObjectNode) sourceNode.get("inputs");
            }
            String targetKey = targetNodeId + "." + targetInputName;
            JsonNode previous = writtenInputs.putIfAbsent(targetKey, renderedValue);
            if (previous != null && !previous.equals(renderedValue)) {
                throw new BusinessException(400, "多个媒体绑定不能覆盖同一个加载输入: " + targetKey);
            }
            nodeInputs.set(targetInputName, renderedValue);
        }
        return workflow;
    }

    private static boolean isUploadedMedia(String valueType) {
        return "uploaded_image".equals(valueType)
                || "uploaded_video".equals(valueType)
                || "uploaded_audio".equals(valueType);
    }

    /**
     * ComfyUI 连线格式：["源节点ID", 槽位索引]。
     */
    private static boolean isLink(JsonNode value) {
        return value != null && value.isArray() && value.size() == 2
                && value.get(0).isTextual() && value.get(1).isInt();
    }

    private Object selectIndexedValue(Object rawValue, ComfyUiInputBinding binding) {
        if (binding.index() == null) {
            if (rawValue instanceof List<?> values
                    && binding.valueType().startsWith("uploaded_")
                    && values.size() == 1) {
                return values.getFirst();
            }
            return rawValue;
        }
        if (!(rawValue instanceof List<?> values) || binding.index() >= values.size()) {
            throw new BusinessException(400,
                    "工作流输入 " + binding.businessField() + " 缺少索引 " + binding.index());
        }
        return values.get(binding.index());
    }

    private JsonNode convertValue(Object value, ComfyUiInputBinding binding) {
        try {
            return switch (binding.valueType()) {
                case "string", "uploaded_image", "uploaded_video", "uploaded_audio" ->
                        objectMapper.getNodeFactory().textNode(requireText(value, binding));
                case "integer" -> objectMapper.getNodeFactory().numberNode(toLong(value, binding));
                case "number" -> objectMapper.getNodeFactory().numberNode(toDouble(value, binding));
                case "boolean" -> objectMapper.getNodeFactory().booleanNode(toBoolean(value, binding));
                case "string_list" -> stringList(value, binding);
                default -> throw new BusinessException(400,
                        "不支持的 ComfyUI 输入绑定类型: " + binding.valueType());
            };
        } catch (NumberFormatException e) {
            throw new BusinessException(400,
                    "工作流输入 " + binding.businessField() + " 不是有效数字");
        }
    }

    private String requireText(Object value, ComfyUiInputBinding binding) {
        String text = value == null ? "" : value.toString();
        if (text.isBlank()) {
            throw new BusinessException(400, "工作流输入不能为空: " + binding.businessField());
        }
        return text;
    }

    private long toLong(Object value, ComfyUiInputBinding binding) {
        if (value instanceof Number number) {
            return number.longValue();
        }
        return Long.parseLong(requireText(value, binding));
    }

    private double toDouble(Object value, ComfyUiInputBinding binding) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        return Double.parseDouble(requireText(value, binding));
    }

    private boolean toBoolean(Object value, ComfyUiInputBinding binding) {
        if (value instanceof Boolean bool) {
            return bool;
        }
        String text = requireText(value, binding);
        if (!"true".equalsIgnoreCase(text) && !"false".equalsIgnoreCase(text)) {
            throw new BusinessException(400,
                    "工作流输入 " + binding.businessField() + " 不是有效布尔值");
        }
        return Boolean.parseBoolean(text);
    }

    private ArrayNode stringList(Object value, ComfyUiInputBinding binding) {
        if (!(value instanceof List<?> values)) {
            throw new BusinessException(400,
                    "工作流输入 " + binding.businessField() + " 必须是列表");
        }
        ArrayNode result = objectMapper.createArrayNode();
        for (Object item : values) {
            if (item == null || item.toString().isBlank()) {
                throw new BusinessException(400,
                        "工作流输入 " + binding.businessField() + " 包含空值");
            }
            result.add(item.toString());
        }
        return result;
    }
}
