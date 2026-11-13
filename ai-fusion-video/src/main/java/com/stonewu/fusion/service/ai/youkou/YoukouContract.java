package com.stonewu.fusion.service.ai.youkou;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** 保存配置时取得的逐模型合同；请求只能使用该端点明确支持的字段。 */
public record YoukouContract(String model, String identity, String kind, JsonNode parameters, JsonNode capabilities) {
    public static final ObjectMapper JSON = new ObjectMapper();
    private static final Map<String, String> LABELS = Map.ofEntries(
            Map.entry("prompt", "提示词"), Map.entry("resolution", "分辨率"),
            Map.entry("aspect_ratio", "画面比例"), Map.entry("size", "图片尺寸"),
            Map.entry("seconds", "视频时长"), Map.entry("quality", "画质"),
            Map.entry("input_images", "参考图片"), Map.entry("input_videos", "参考视频"),
            Map.entry("input_audios", "参考音频"), Map.entry("generate_audio", "生成声音"));

    public static YoukouContract from(AiModel model) {
        ObjectNode config = object(model.getConfig(), "模型配置无法读取，请重新添加贰春模型");
        JsonNode contract = config.path("youkouContract");
        if (!model.getCode().equals(contract.path("model").asText())
                || !contract.path("publicModelId").asText().matches("mdl_[a-f0-9]{32}")
                || !contract.path("parameters").isObject() || !contract.path("capabilities").isObject()) {
            throw new BusinessException("贰春模型能力尚未配置，请从平台重新读取模型");
        }
        String kind = switch (model.getModelType()) {
            case 1 -> "text";
            case 2 -> "image";
            case 3 -> "video";
            default -> throw new BusinessException("贰春当前只接入对话、图片和视频生成");
        };
        if (!kind.equals(contract.path("kind").asText())) {
            throw new BusinessException("模型类型与平台目录不一致，请重新读取模型");
        }
        return new YoukouContract(model.getCode(), contract.path("publicModelId").asText(), kind,
                contract.path("parameters"), contract.path("capabilities"));
    }

    public JsonNode fields(String endpoint) {
        JsonNode operation = parameters.path(endpoint);
        if (!operation.path("fields").isObject()
                || !values(operation.path("content_types")).contains("application/json")) {
            throw new BusinessException("当前模型未提供所需的生成接口，请重新读取模型");
        }
        return operation.path("fields");
    }

    public boolean supports(JsonNode fields, String field) {
        return "supported".equals(fields.path(field).path("status").asText());
    }

    public void put(ObjectNode body, JsonNode fields, String name, Object value) {
        if (value != null && (!(value instanceof String text) || !text.isBlank())) {
            body.set(name, JSON.valueToTree(value));
        }
    }

    public void defaults(ObjectNode body, JsonNode fields) {
        // 别名、素材和模型身份由请求构造器处理，不能把目录默认值重复拼入请求。
        Set<String> excluded = Set.of("model", "prompt", "size", "input_images", "input_videos", "input_audios",
                "input_reference", "input_image_files", "image", "generation_options");
        fields.properties().forEach(entry -> {
            String name = entry.getKey();
            JsonNode field = entry.getValue();
            JsonNode value = field.path("default");
            if (!excluded.contains(name) && !body.has(name) && supports(fields, name)
                    && !field.has("alias_of") && !value.isMissingNode() && !value.isNull()
                    && value.isValueNode()) {
                body.set(name, value.deepCopy());
            }
        });
    }

    public void validate(ObjectNode body, JsonNode fields) {
        validateValues(body, fields);
        fields.properties().forEach(entry -> {
            String name = entry.getKey();
            JsonNode field = entry.getValue();
            if (field.path("required").asBoolean() && !present(body.path(name))) {
                throw invalid(name, "请填写或上传后再生成");
            }
            List<String> alternatives = values(field.path("required_any_of"));
            if (!alternatives.isEmpty() && alternatives.stream().noneMatch(key -> present(body.path(key)))) {
                throw new BusinessException("请添加当前模型要求的参考素材");
            }
            if (body.has(name)) {
                if (values(field.path("exclusive_with")).stream().anyMatch(body::has)) {
                    throw invalid(name, "与其他参数冲突，请只选择一种输入方式");
                }
                if (values(field.path("requires")).stream().anyMatch(key -> !present(body.path(key)))) {
                    throw invalid(name, "缺少所需的关联参数");
                }
            }
        });
    }

    public void validateValues(ObjectNode body, JsonNode fields) {
        body.properties().forEach(entry -> validateValue(entry.getKey(), entry.getValue(), fields.path(entry.getKey())));
    }

    private void validateValue(String name, JsonNode value, JsonNode field) {
        if (!"supported".equals(field.path("status").asText())) {
            throw invalid(name, "当前模型尚未开放此参数，请移除后再生成");
        }
        boolean typeMatches = switch (field.path("type").asText()) {
            case "string" -> value.isTextual();
            case "integer" -> value.isIntegralNumber();
            case "number" -> value.isNumber();
            case "boolean" -> value.isBoolean();
            case "array" -> value.isArray();
            case "object" -> value.isObject();
            default -> false;
        };
        if (!typeMatches) {
            throw invalid(name, "格式不符合模型要求");
        }
        JsonNode allowed = field.path("values");
        if (allowed.isArray() && !allowed.isEmpty()
                && !allowed.valueStream().anyMatch(candidate -> candidate.equals(value))) {
            throw invalid(name, "不在模型支持的范围内，请重新选择");
        }
        if (value.isNumber()) {
            BigDecimal number = value.decimalValue();
            if (field.path("min").isNumber() && number.compareTo(field.get("min").decimalValue()) < 0
                    || field.path("max").isNumber() && number.compareTo(field.get("max").decimalValue()) > 0) {
                throw invalid(name, "超出模型允许范围");
            }
        }
        if (value.isTextual()) {
            String text = value.textValue();
            if (field.path("max_length").asInt() > 0
                    && text.codePointCount(0, text.length()) > field.get("max_length").asInt()
                    || field.path("max_bytes").asLong() > 0
                    && text.getBytes(StandardCharsets.UTF_8).length > field.get("max_bytes").asLong()) {
                throw invalid(name, "内容过长，请缩短后再生成");
            }
        }
        if (value.isArray()) {
            if (value.size() < field.path("min_count").asInt()
                    || field.has("max_count") && value.size() > field.get("max_count").asInt()) {
                throw invalid(name, "数量不符合模型要求");
            }
            if ("media_id".equals(field.path("format").asText())) {
                for (JsonNode item : value) {
                    if (!item.isObject() || item.size() != 1
                            || !item.path("media_id").asText().matches("med_[a-f0-9]{32}")) {
                        throw invalid(name, "素材尚未上传成功，请重新添加");
                    }
                }
            }
        }
        if (value.isObject()) {
            JsonNode nested = field.path("options").path("fields");
            if (!nested.isObject()) {
                if (!value.isEmpty()) {
                    throw invalid(name, "平台尚未公布此选项的完整规则");
                }
            } else {
                validate((ObjectNode) value, nested);
            }
        }
    }

    public static ObjectNode object(String text, String message) {
        try {
            JsonNode node = JSON.readTree(text == null ? "" : text);
            if (node instanceof ObjectNode object) {
                return object;
            }
        } catch (IOException invalid) {
            throw new BusinessException(message);
        }
        throw new BusinessException(message);
    }

    public static List<String> values(JsonNode node) {
        return node.isArray() ? node.valueStream().map(JsonNode::asText).toList() : List.of();
    }

    private static boolean present(JsonNode node) {
        return !node.isMissingNode() && !node.isNull()
                && (!node.isTextual() || !node.asText().isBlank()) && (!node.isArray() || !node.isEmpty());
    }

    public static BusinessException invalid(String name, String detail) {
        return new BusinessException(LABELS.getOrDefault(name, "模型选项") + "：" + detail);
    }
}
