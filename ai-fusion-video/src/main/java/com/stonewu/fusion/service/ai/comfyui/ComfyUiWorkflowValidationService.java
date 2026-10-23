package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiConnectionRespVO;
import com.stonewu.fusion.controller.ai.vo.comfyui.ComfyUiWorkflowValidationRespVO;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ProviderConfigurationFingerprint;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiConnectionResult;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiNativeClient;
import com.stonewu.fusion.service.canvas.CanvasJson;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * Online validation against the exact target ComfyUI instance.
 */
@Service
@RequiredArgsConstructor
public class ComfyUiWorkflowValidationService {

    private final ApiConfigService apiConfigService;
    private final ComfyUiWorkflowService workflowService;
    private final ComfyUiWorkflowDocumentService documentService;
    private final ComfyUiNativeClient nativeClient;
    private final CanvasJson json;

    public ComfyUiConnectionRespVO testConnection(Long apiConfigId) {
        ApiConfig apiConfig = requireComfyUiApiConfig(apiConfigId);
        ComfyUiConnectionResult result = nativeClient.testConnection(apiConfig);
        return ComfyUiConnectionRespVO.builder()
                .connected(result.connected())
                .jobsApiSupported(result.jobsApiSupported())
                .version(result.version())
                .systemStats(result.systemStats())
                .features(result.features())
                .build();
    }

    public ComfyUiWorkflowValidationRespVO validateVersion(Long versionId) {
        ComfyUiWorkflowVersion version = workflowService.requireVersion(versionId);
        ComfyUiWorkflow workflow = workflowService.requireWorkflow(version.getWorkflowId());
        ApiConfig apiConfig = requireComfyUiApiConfig(workflow.getApiConfigId());
        String apiHash = ProviderConfigurationFingerprint.of(apiConfig, json);
        try {
            nativeClient.testConnection(apiConfig);
            ObjectNode apiWorkflow = documentService.parseApiWorkflow(version.getApiWorkflowJson());
            Set<String> missingClasses = new LinkedHashSet<>();
            List<String> invalidModelInputs = new ArrayList<>();
            Map<String, JsonNode> definitionCache = new HashMap<>();
            Iterator<Map.Entry<String, JsonNode>> nodes = apiWorkflow.fields();
            int checkedNodes = 0;
            while (nodes.hasNext()) {
                Map.Entry<String, JsonNode> entry = nodes.next();
                checkedNodes++;
                String nodeId = entry.getKey();
                JsonNode workflowNode = entry.getValue();
                String classType = workflowNode.path("class_type").asText();
                JsonNode definition = definitionCache.computeIfAbsent(classType,
                        ignored -> nativeClient.getNodeInfo(apiConfig, classType).path(classType));
                if (definition.isMissingNode() || !definition.isObject()) {
                    missingClasses.add(classType);
                    continue;
                }
                validateChoiceInputs(nodeId, workflowNode.path("inputs"), definition, invalidModelInputs);
            }
            boolean valid = missingClasses.isEmpty() && invalidModelInputs.isEmpty();
            String message = valid
                    ? "已在目标 ComfyUI 校验 " + checkedNodes + " 个节点"
                    : buildFailureMessage(missingClasses, invalidModelInputs);
            workflowService.recordValidation(versionId, version.getWorkflowHash(), apiHash, valid, message);
            return ComfyUiWorkflowValidationRespVO.builder()
                    .valid(valid)
                    .checkedNodeCount(checkedNodes)
                    .missingNodeClasses(List.copyOf(missingClasses))
                    .invalidModelInputs(List.copyOf(invalidModelInputs))
                    .message(message)
                    .build();
        } catch (BusinessException error) {
            workflowService.recordValidation(versionId, version.getWorkflowHash(), apiHash, false, error.getMessage());
            throw error;
        }
    }

    private void validateChoiceInputs(String nodeId,
                                      JsonNode workflowInputs,
                                      JsonNode definition,
                                      List<String> invalidInputs) {
        JsonNode definitions = definition.path("input");
        Iterator<Map.Entry<String, JsonNode>> inputs = workflowInputs.fields();
        while (inputs.hasNext()) {
            Map.Entry<String, JsonNode> input = inputs.next();
            if (!input.getValue().isTextual()) {
                continue;
            }
            JsonNode inputDefinition = definitions.path("required").get(input.getKey());
            if (inputDefinition == null) {
                inputDefinition = definitions.path("optional").get(input.getKey());
            }
            if (inputDefinition == null || !inputDefinition.isArray()
                    || inputDefinition.isEmpty() || !inputDefinition.get(0).isArray()) {
                continue;
            }
            boolean allowed = false;
            for (JsonNode choice : inputDefinition.get(0)) {
                if (choice.isTextual() && choice.asText().equals(input.getValue().asText())) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed) {
                invalidInputs.add(nodeId + "." + input.getKey() + "=" + input.getValue().asText());
            }
        }
    }

    private String buildFailureMessage(Set<String> missingClasses, List<String> invalidInputs) {
        List<String> parts = new ArrayList<>();
        if (!missingClasses.isEmpty()) {
            parts.add("缺少节点: " + String.join(", ", missingClasses));
        }
        if (!invalidInputs.isEmpty()) {
            parts.add("目标实例不存在模型/选项: " + String.join(", ", invalidInputs));
        }
        return String.join("；", parts);
    }

    private ApiConfig requireComfyUiApiConfig(Long id) {
        ApiConfig apiConfig = id == null ? null : apiConfigService.getById(id);
        if (apiConfig == null) {
            throw new BusinessException(404, "API 配置不存在");
        }
        if (!ComfyUiWorkflowService.PLATFORM.equalsIgnoreCase(apiConfig.getPlatform())) {
            throw new BusinessException(400, "API 配置不是 ComfyUI 平台");
        }
        if (!Integer.valueOf(1).equals(apiConfig.getStatus())) {
            throw new BusinessException(400, "ComfyUI API 配置已禁用");
        }
        return apiConfig;
    }
}
