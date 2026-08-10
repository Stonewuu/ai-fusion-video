package com.stonewu.fusion.service.ai.model;

import cn.hutool.core.util.StrUtil;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;

import java.util.List;
import java.util.Objects;

/** Builds immutable, request-local model options without mutating the persisted model. */
public final class AiModelRequestOptions {

    private static final String REASONING_EFFORT = "reasoningEffort";

    private AiModelRequestOptions() {
    }

    public static AiModel withReasoningEffort(
            AiModel source,
            String requestedEffort,
            ObjectMapper objectMapper) {
        AiModel safeSource = Objects.requireNonNull(source, "source must not be null");
        ObjectMapper mapper = Objects.requireNonNull(objectMapper, "objectMapper must not be null");
        String effort = normalize(requestedEffort);
        if (effort == null) {
            return safeSource;
        }
        if (!Boolean.TRUE.equals(safeSource.getSupportReasoning())) {
            throw new BusinessException(400, "当前模型不支持思考等级");
        }
        List<String> levels = safeSource.getReasoningEffortLevels() == null
                ? List.of()
                : safeSource.getReasoningEffortLevels();
        if (!levels.contains(effort)) {
            throw new BusinessException(400, "模型未配置思考等级: " + effort);
        }

        ObjectNode config = parseConfig(safeSource.getConfig(), mapper);
        config.put(REASONING_EFFORT, effort);
        try {
            return copyWithConfig(safeSource, mapper.writeValueAsString(config));
        } catch (JsonProcessingException e) {
            throw new BusinessException(400, "模型配置 JSON 无法序列化");
        }
    }

    public static String reasoningEffort(JsonNode modelOptions) {
        if (modelOptions == null || !modelOptions.isObject()) {
            return null;
        }
        JsonNode value = modelOptions.get(REASONING_EFFORT);
        return value != null && value.isTextual() ? normalize(value.asText()) : null;
    }

    private static ObjectNode parseConfig(String configJson, ObjectMapper objectMapper) {
        if (StrUtil.isBlank(configJson)) {
            return objectMapper.createObjectNode();
        }
        try {
            JsonNode parsed = objectMapper.readTree(configJson);
            if (parsed == null || !parsed.isObject()) {
                throw new BusinessException(400, "模型配置必须是 JSON 对象");
            }
            return ((ObjectNode) parsed).deepCopy();
        } catch (JsonProcessingException e) {
            throw new BusinessException(400, "模型配置 JSON 无效");
        }
    }

    private static String normalize(String value) {
        return StrUtil.isBlank(value) ? null : value.trim();
    }

    private static AiModel copyWithConfig(AiModel source, String config) {
        return AiModel.builder()
                .id(source.getId())
                .name(source.getName())
                .code(source.getCode())
                .modelProtocol(source.getModelProtocol())
                .capabilityPresetCode(source.getCapabilityPresetCode())
                .modelType(source.getModelType())
                .icon(source.getIcon())
                .description(source.getDescription())
                .sort(source.getSort())
                .status(source.getStatus())
                .config(config)
                .maxConcurrency(source.getMaxConcurrency())
                .apiConfigId(source.getApiConfigId())
                .defaultModel(source.getDefaultModel())
                .supportVision(source.getSupportVision())
                .multimodalInputTypes(source.getMultimodalInputTypes())
                .multimodalInputTransports(source.getMultimodalInputTransports())
                .supportReasoning(source.getSupportReasoning())
                .reasoningEffortLevels(source.getReasoningEffortLevels())
                .contextWindow(source.getContextWindow())
                .deletedId(source.getDeletedId())
                .build();
    }
}
