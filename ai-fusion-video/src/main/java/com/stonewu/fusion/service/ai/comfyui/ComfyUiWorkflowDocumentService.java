package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** Parses, canonicalizes and validates ComfyUI workflow documents without network access. */
@Service
@RequiredArgsConstructor
public class ComfyUiWorkflowDocumentService {

    private static final int MAX_API_JSON_BYTES = 2 * 1024 * 1024;
    private static final int MAX_UI_JSON_BYTES = 4 * 1024 * 1024;
    private static final int MAX_NODES = 500;
    private static final int MAX_BINDINGS = 128;
    private static final Set<String> COMMON_FIELDS = Set.of(
            "prompt", "negativePrompt", "seed", "count");
    private static final Set<String> IMAGE_FIELDS = Set.of(
            "width", "height", "referenceImages");
    private static final Set<String> VIDEO_FIELDS = Set.of(
            "width", "height", "duration", "fps", "firstFrame", "lastFrame",
            "referenceImages", "referenceVideos", "referenceAudios", "generateAudio");
    private static final Set<String> VALUE_TYPES = Set.of(
            "string", "integer", "number", "boolean", "string_list",
            "uploaded_image", "uploaded_video", "uploaded_audio");
    private static final Set<String> OUTPUT_MEDIA_TYPES = Set.of(
            "image", "video", "audio", "file");
    private static final Set<String> OUTPUT_ROLES = Set.of(
            "primary", "cover", "auxiliary");
    private static final Set<String> SECRET_FIELD_NAMES = Set.of(
            "api_key", "apikey", "token", "password", "secret", "authorization");

    private final ObjectMapper objectMapper;

    public ComfyUiWorkflowDefinition normalize(int modelType,
                                                String uiWorkflowJson,
                                                String apiWorkflowJson,
                                                String inputBindingsJson,
                                                String outputBindingsJson) {
        requireModelType(modelType);
        ObjectNode apiWorkflow = parseObject(apiWorkflowJson, "API-format 工作流", MAX_API_JSON_BYTES, true);
        if (apiWorkflow.has("nodes") || apiWorkflow.has("links")) {
            throw invalid("检测到 UI-format 工作流，请从 ComfyUI 使用 Export (API) 导出");
        }
        if (apiWorkflow.isEmpty()) {
            throw invalid("API-format 工作流不能为空");
        }
        if (apiWorkflow.size() > MAX_NODES) {
            throw invalid("工作流节点数量不能超过 " + MAX_NODES);
        }

        Map<String, String> nodeTypes = validateNodes(apiWorkflow);
        rejectEmbeddedSecrets(apiWorkflow, "prompt");
        ObjectNode inputBindingsNode = parseObject(
                defaultJson(inputBindingsJson, "{}"), "输入绑定", MAX_API_JSON_BYTES, false);
        ArrayNode outputBindingsNode = parseArray(
                defaultJson(outputBindingsJson, "[]"), "输出绑定", MAX_API_JSON_BYTES);
        List<ComfyUiInputBinding> inputBindings = validateInputBindings(
                modelType, apiWorkflow, inputBindingsNode);
        List<ComfyUiOutputBinding> outputBindings = validateOutputBindings(
                modelType, apiWorkflow, outputBindingsNode);

        String canonicalApi = writeCanonical(apiWorkflow);
        String canonicalInputs = writeCanonical(inputBindingsNode);
        String canonicalOutputs = writeCanonical(outputBindingsNode);
        String canonicalUi = null;
        if (uiWorkflowJson != null && !uiWorkflowJson.isBlank()) {
            canonicalUi = writeCanonical(parseObject(
                    uiWorkflowJson, "UI-format 工作流", MAX_UI_JSON_BYTES, false));
        }
        List<String> requiredNodes = nodeTypes.values().stream().distinct().sorted().toList();
        String requiredNodesJson = writeCanonical(objectMapper.valueToTree(requiredNodes));
        String hash = sha256(canonicalApi + "\n" + canonicalInputs + "\n" + canonicalOutputs);
        return new ComfyUiWorkflowDefinition(
                canonicalUi,
                canonicalApi,
                canonicalInputs,
                canonicalOutputs,
                requiredNodesJson,
                hash,
                List.copyOf(inputBindings),
                List.copyOf(outputBindings));
    }

    public ObjectNode parseApiWorkflow(String json) {
        return parseObject(json, "API-format 工作流", MAX_API_JSON_BYTES, true);
    }

    public List<ComfyUiInputBinding> parseInputBindings(int modelType,
                                                        String apiWorkflowJson,
                                                        String bindingsJson) {
        ObjectNode workflow = parseApiWorkflow(apiWorkflowJson);
        return validateInputBindings(modelType, workflow,
                parseObject(defaultJson(bindingsJson, "{}"), "输入绑定", MAX_API_JSON_BYTES, false));
    }

    public List<ComfyUiOutputBinding> parseOutputBindings(int modelType,
                                                          String apiWorkflowJson,
                                                          String bindingsJson) {
        ObjectNode workflow = parseApiWorkflow(apiWorkflowJson);
        return validateOutputBindings(modelType, workflow,
                parseArray(defaultJson(bindingsJson, "[]"), "输出绑定", MAX_API_JSON_BYTES));
    }

    private Map<String, String> validateNodes(ObjectNode workflow) {
        Map<String, String> nodeTypes = new LinkedHashMap<>();
        Iterator<Map.Entry<String, JsonNode>> fields = workflow.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> entry = fields.next();
            String nodeId = entry.getKey();
            JsonNode node = entry.getValue();
            if (nodeId.isBlank() || !node.isObject()) {
                throw invalid("工作流节点必须使用非空 ID 和对象结构: " + nodeId);
            }
            String classType = text(node.get("class_type"));
            if (classType == null) {
                throw invalid("工作流节点缺少 class_type: " + nodeId);
            }
            if (!node.path("inputs").isObject()) {
                throw invalid("工作流节点 inputs 必须是对象: " + nodeId);
            }
            nodeTypes.put(nodeId, classType);
        }
        return nodeTypes;
    }

    private List<ComfyUiInputBinding> validateInputBindings(int modelType,
                                                            ObjectNode workflow,
                                                            ObjectNode bindings) {
        Set<String> allowedFields = new HashSet<>(COMMON_FIELDS);
        allowedFields.addAll(modelType == 2 ? IMAGE_FIELDS : VIDEO_FIELDS);
        List<ComfyUiInputBinding> resolved = new ArrayList<>();
        Set<String> targets = new LinkedHashSet<>();
        Iterator<Map.Entry<String, JsonNode>> fields = bindings.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> field = fields.next();
            String businessField = field.getKey();
            if (!allowedFields.contains(businessField)) {
                throw invalid("不支持的工作流业务输入: " + businessField);
            }
            if (!field.getValue().isArray() || field.getValue().isEmpty()) {
                throw invalid("输入绑定必须是非空数组: " + businessField);
            }
            for (JsonNode binding : field.getValue()) {
                if (!binding.isObject()) {
                    throw invalid("输入绑定项必须是对象: " + businessField);
                }
                String nodeId = requiredText(binding, "nodeId", "输入绑定缺少 nodeId");
                String inputName = requiredText(binding, "inputName", "输入绑定缺少 inputName");
                String valueType = requiredText(binding, "valueType", "输入绑定缺少 valueType")
                        .toLowerCase(Locale.ROOT);
                Integer index = binding.hasNonNull("index") ? binding.get("index").asInt() : null;
                if (!VALUE_TYPES.contains(valueType)) {
                    throw invalid("不支持的输入绑定 valueType: " + valueType);
                }
                JsonNode node = workflow.get(nodeId);
                if (node == null) {
                    throw invalid("输入绑定引用不存在的节点: " + nodeId);
                }
                if (!node.path("inputs").has(inputName)) {
                    throw invalid("节点 " + nodeId + " 不包含输入字段: " + inputName);
                }
                if (index != null && index < 0) {
                    throw invalid("输入绑定 index 不能小于 0");
                }
                String targetKey = nodeId + "\u0000" + inputName;
                if (!targets.add(targetKey)) {
                    throw invalid("同一个节点输入不能重复绑定: " + nodeId + "." + inputName);
                }
                resolved.add(new ComfyUiInputBinding(
                        businessField, nodeId, inputName, valueType, index));
                if (resolved.size() > MAX_BINDINGS) {
                    throw invalid("输入绑定数量不能超过 " + MAX_BINDINGS);
                }
            }
        }
        return resolved;
    }

    private List<ComfyUiOutputBinding> validateOutputBindings(int modelType,
                                                              ObjectNode workflow,
                                                              ArrayNode bindings) {
        if (bindings.isEmpty()) {
            throw invalid("至少需要配置一个输出绑定");
        }
        List<ComfyUiOutputBinding> resolved = new ArrayList<>();
        Set<String> unique = new HashSet<>();
        boolean hasPrimaryTarget = false;
        String requiredPrimaryType = modelType == 2 ? "image" : "video";
        for (JsonNode binding : bindings) {
            if (!binding.isObject()) {
                throw invalid("输出绑定项必须是对象");
            }
            String nodeId = requiredText(binding, "nodeId", "输出绑定缺少 nodeId");
            String mediaType = requiredText(binding, "mediaType", "输出绑定缺少 mediaType")
                    .toLowerCase(Locale.ROOT);
            String role = requiredText(binding, "role", "输出绑定缺少 role")
                    .toLowerCase(Locale.ROOT);
            if (workflow.get(nodeId) == null) {
                throw invalid("输出绑定引用不存在的节点: " + nodeId);
            }
            if (!OUTPUT_MEDIA_TYPES.contains(mediaType)) {
                throw invalid("不支持的输出媒体类型: " + mediaType);
            }
            if (!OUTPUT_ROLES.contains(role)) {
                throw invalid("不支持的输出角色: " + role);
            }
            if (!unique.add(nodeId + "\u0000" + mediaType + "\u0000" + role)) {
                throw invalid("输出绑定重复: " + nodeId);
            }
            if ("primary".equals(role) && requiredPrimaryType.equals(mediaType)) {
                hasPrimaryTarget = true;
            }
            resolved.add(new ComfyUiOutputBinding(nodeId, mediaType, role));
            if (resolved.size() > MAX_BINDINGS) {
                throw invalid("输出绑定数量不能超过 " + MAX_BINDINGS);
            }
        }
        if (!hasPrimaryTarget) {
            throw invalid("必须配置一个 primary " + requiredPrimaryType + " 输出");
        }
        return resolved;
    }

    private void rejectEmbeddedSecrets(JsonNode node, String path) {
        if (node.isObject()) {
            Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> field = fields.next();
                String normalized = field.getKey().trim().toLowerCase(Locale.ROOT).replace('-', '_');
                if (SECRET_FIELD_NAMES.contains(normalized)
                        && field.getValue().isTextual()
                        && !field.getValue().asText().isBlank()) {
                    throw invalid("工作流中不能保存明文密钥字段: " + path + "." + field.getKey());
                }
                rejectEmbeddedSecrets(field.getValue(), path + "." + field.getKey());
            }
        } else if (node.isArray()) {
            for (int i = 0; i < node.size(); i++) {
                rejectEmbeddedSecrets(node.get(i), path + "[" + i + "]");
            }
        }
    }

    private ObjectNode parseObject(String json, String label, int maxBytes, boolean required) {
        String value = json == null ? "" : json.trim();
        if (value.isEmpty()) {
            if (required) {
                throw invalid(label + "不能为空");
            }
            value = "{}";
        }
        checkSize(value, label, maxBytes);
        try {
            JsonNode parsed = objectMapper.readTree(value);
            if (!(parsed instanceof ObjectNode objectNode)) {
                throw invalid(label + "必须是 JSON 对象");
            }
            return objectNode;
        } catch (JsonProcessingException e) {
            throw invalid(label + "不是合法 JSON: " + e.getOriginalMessage());
        }
    }

    private ArrayNode parseArray(String json, String label, int maxBytes) {
        String value = json == null ? "" : json.trim();
        checkSize(value, label, maxBytes);
        try {
            JsonNode parsed = objectMapper.readTree(value);
            if (!(parsed instanceof ArrayNode arrayNode)) {
                throw invalid(label + "必须是 JSON 数组");
            }
            return arrayNode;
        } catch (JsonProcessingException e) {
            throw invalid(label + "不是合法 JSON: " + e.getOriginalMessage());
        }
    }

    private String writeCanonical(JsonNode node) {
        try {
            return objectMapper.writeValueAsString(sortRecursively(node));
        } catch (JsonProcessingException e) {
            throw new BusinessException(500, "序列化 ComfyUI 工作流失败");
        }
    }

    private JsonNode sortRecursively(JsonNode node) {
        if (node.isObject()) {
            ObjectNode sorted = objectMapper.createObjectNode();
            List<String> names = new ArrayList<>();
            node.fieldNames().forEachRemaining(names::add);
            names.sort(Comparator.naturalOrder());
            for (String name : names) {
                sorted.set(name, sortRecursively(node.get(name)));
            }
            return sorted;
        }
        if (node.isArray()) {
            ArrayNode array = objectMapper.createArrayNode();
            node.forEach(item -> array.add(sortRecursively(item)));
            return array;
        }
        return node.deepCopy();
    }

    private String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256")
                    .digest(value.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }

    private void checkSize(String value, String label, int maxBytes) {
        if (value.getBytes(StandardCharsets.UTF_8).length > maxBytes) {
            throw invalid(label + "超过大小限制");
        }
    }

    private String requiredText(JsonNode node, String field, String message) {
        String value = text(node.get(field));
        if (value == null) {
            throw invalid(message);
        }
        return value;
    }

    private String text(JsonNode node) {
        if (node == null || !node.isTextual() || node.asText().isBlank()) {
            return null;
        }
        return node.asText().trim();
    }

    private String defaultJson(String value, String fallback) {
        return value == null || value.isBlank() ? fallback : value;
    }

    private void requireModelType(int modelType) {
        if (modelType != 2 && modelType != 3) {
            throw invalid("ComfyUI 工作流仅支持图片或视频模型");
        }
    }

    private BusinessException invalid(String message) {
        return new BusinessException(400, message);
    }
}
