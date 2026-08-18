package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/** Applies explicit platform bindings to a deep copy of an immutable workflow version. */
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
            ObjectNode node = (ObjectNode) workflow.get(binding.nodeId());
            ObjectNode nodeInputs = (ObjectNode) node.get("inputs");
            JsonNode current = nodeInputs.get(binding.inputName());
            if (isUploadedMedia(binding.valueType()) && isLink(current)) {
                // IMAGE 类输入需要张量而非文件名：目标输入当前是连线（如 LoadImage → 节点）时，
                // 把上传文件名写到连线源节点的 image 输入，由 ComfyUI 加载成张量后经连线传入；
                // 直接覆盖目标输入会把字符串塞进 IMAGE 输入导致 TypeError。
                String sourceNodeId = current.get(0).asText();
                ObjectNode sourceNode = (ObjectNode) workflow.get(sourceNodeId);
                if (sourceNode != null && sourceNode.path("inputs").isObject()
                        && sourceNode.path("inputs").has("image")) {
                    ((ObjectNode) sourceNode.get("inputs")).set("image", renderedValue);
                    continue;
                }
            }
            nodeInputs.set(binding.inputName(), renderedValue);
        }
        return workflow;
    }

    private static boolean isUploadedMedia(String valueType) {
        return "uploaded_image".equals(valueType)
                || "uploaded_video".equals(valueType)
                || "uploaded_audio".equals(valueType);
    }

    /** ComfyUI 连线格式：["源节点ID", 槽位索引]。 */
    private static boolean isLink(JsonNode value) {
        return value != null && value.isArray() && value.size() >= 2
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
        if (value instanceof Number number) return number.longValue();
        return Long.parseLong(requireText(value, binding));
    }

    private double toDouble(Object value, ComfyUiInputBinding binding) {
        if (value instanceof Number number) return number.doubleValue();
        return Double.parseDouble(requireText(value, binding));
    }

    private boolean toBoolean(Object value, ComfyUiInputBinding binding) {
        if (value instanceof Boolean bool) return bool;
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
