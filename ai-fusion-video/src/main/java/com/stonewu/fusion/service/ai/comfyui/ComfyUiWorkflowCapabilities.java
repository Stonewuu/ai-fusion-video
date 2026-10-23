package com.stonewu.fusion.service.ai.comfyui;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeSet;
import java.math.BigDecimal;

/** 已发布工作流提供输入能力，模型配置只能收紧使用范围，不回写模型或工作流。 */
@Component
@RequiredArgsConstructor
public class ComfyUiWorkflowCapabilities {
    public static final String VERSION_KEY = "comfyuiWorkflowVersionId";
    public static final String INPUTS_KEY = "comfyuiInputs";
    public static final String LIMITS_KEY = "comfyuiInputLimits";
    private static final List<String> MEDIA_FIELDS = List.of(
            "referenceImages", "firstFrame", "lastFrame", "referenceVideos", "referenceAudios");
    private static final Map<String, String> MAXIMUM_KEYS = Map.of(
            "referenceImages", "maxReferenceImages", "firstFrame", "maxFirstFrames",
            "lastFrame", "maxLastFrames", "referenceVideos", "maxReferenceVideos",
            "referenceAudios", "maxReferenceAudios", "imageInputs", "maxImageInputs");
    private static final Map<String, String> SUPPORT_KEYS = Map.of(
            "referenceImages", "supportReferenceImages", "firstFrame", "supportFirstFrame",
            "lastFrame", "supportLastFrame", "referenceVideos", "supportReferenceVideos",
            "referenceAudios", "supportReferenceAudios");
    private static final Map<String, List<String>> LEGACY_SUPPORT_KEYS = Map.of(
            "referenceImages", List.of("supportsReferenceImages", "supportRefImages", "supportImageReferences"),
            "firstFrame", List.of("supportsFirstFrame", "allowFirstFrame", "supportStartImage"),
            "lastFrame", List.of("supportsLastFrame", "allowLastFrame", "supportEndImage"),
            "referenceVideos", List.of("supportsReferenceVideos", "supportRefVideos", "supportVideoReferences"),
            "referenceAudios", List.of("supportsReferenceAudios", "supportRefAudios", "supportAudioReferences"));

    private final ComfyUiWorkflowService workflows;
    private final ComfyUiWorkflowDocumentService documents;

    public JSONObject resolve(AiModel model, JSONObject configured, Long requestedVersion) {
        var workflow = workflows.requireWorkflow(model.getComfyuiWorkflowId());
        if (!Integer.valueOf(1).equals(workflow.getStatus())
                || !Objects.equals(workflow.getApiConfigId(), model.getApiConfigId())
                || !Objects.equals(workflow.getModelType(), model.getModelType())) {
            throw invalid("工作流已停用，或与模型的类型、API 配置不匹配");
        }
        Long versionId = requestedVersion == null ? workflow.getActiveVersionId() : requestedVersion;
        if (versionId == null) {
            throw invalid("工作流没有已发布版本");
        }
        var version = workflows.requireVersion(versionId);
        if (!Boolean.TRUE.equals(version.getPublished()) || !Objects.equals(workflow.getId(), version.getWorkflowId())) {
            throw invalid("工作流版本未发布或不属于当前模型");
        }
        return describe(model, version, configured);
    }

    public JSONObject describe(AiModel model, ComfyUiWorkflowVersion version, JSONObject configured) {
        var bindings = documents.parseInputBindings(model.getModelType(), version.getApiWorkflowJson(),
                version.getInputBindingsJson());
        var fields = bindings.stream().map(ComfyUiInputBinding::businessField).distinct().toList();
        JSONObject result = JSONUtil.parseObj(configured.toString());
        if (configured.get(LIMITS_KEY) != null && !JSONUtil.isNull(configured.get(LIMITS_KEY))
                && !(configured.get(LIMITS_KEY) instanceof JSONObject)) {
            throw invalid("输入限制必须是 JSON 对象");
        }
        JSONObject limits = configured.getJSONObject(LIMITS_KEY);
        validateLimitKeys(limits);
        JSONObject inputs = new JSONObject();
        for (String field : fields) {
            var matching = bindings.stream().filter(binding -> field.equals(binding.businessField())).toList();
            if (matching.stream().anyMatch(binding -> "uploaded_video".equals(binding.valueType())
                    || "uploaded_audio".equals(binding.valueType()))) {
                throw invalid("当前接入不支持视频或音频文件上传绑定：" + field);
            }
            var indices = new TreeSet<Integer>();
            boolean wholeList = false;
            for (var binding : matching) {
                if (binding.index() == null && "string_list".equals(binding.valueType())) {
                    wholeList = true;
                } else {
                    indices.add(binding.index() == null ? 0 : binding.index());
                }
            }
            int minimumWhenProvided = indices.isEmpty() ? 0 : indices.last() + 1;
            if (List.of("firstFrame", "lastFrame").contains(field) && minimumWhenProvided > 1) {
                throw invalid(inputLabel(field) + " 只接收一张图片，不能绑定多个索引");
            }
            if (!wholeList && indices.size() != minimumWhenProvided) {
                throw invalid("输入 " + inputLabel(field) + " 的索引必须从 0 连续绑定，不能遗漏素材");
            }
            Integer maximum = wholeList ? null : minimumWhenProvided;
            Integer usageLimit = MEDIA_FIELDS.contains(field) ? limit(configured, limits, field) : null;
            if (usageLimit != null) {
                maximum = maximum == null ? usageLimit : Math.min(maximum, usageLimit);
            }
            if (maximum != null && maximum > 0 && maximum < minimumWhenProvided) {
                throw invalid("输入 " + inputLabel(field) + " 的使用上限 " + maximum + " 小于工作流所需的 "
                        + minimumWhenProvided + " 项，请调整限制或工作流");
            }
            inputs.set(field, new JSONObject().set("maximum", maximum).set("minimum", 0)
                    .set("minimumWhenProvided", minimumWhenProvided)
                    .set("workflowMaximum", wholeList ? null : minimumWhenProvided)
                    .set("usageMaximum", usageLimit));
        }
        Integer totalLimit = limit(configured, limits, "imageInputs");
        if (totalLimit != null) {
            for (String field : List.of("firstFrame", "lastFrame", "referenceImages")) {
                JSONObject input = inputs.getJSONObject(field);
                if (input != null) {
                    Integer maximum = input.getInt("maximum");
                    int limited = maximum == null ? totalLimit : Math.min(maximum, totalLimit);
                    if (limited > 0 && limited < input.getInt("minimumWhenProvided", 0)) {
                        throw invalid("图片输入总数上限小于工作流输入 " + inputLabel(field) + " 所需数量");
                    }
                    input.set("maximum", limited);
                }
            }
        }
        for (String field : MEDIA_FIELDS) {
            JSONObject input = inputs.getJSONObject(field);
            Integer maximum = input == null ? Integer.valueOf(0) : input.getInt("maximum");
            result.set(SUPPORT_KEYS.get(field), input != null && (maximum == null || maximum > 0));
            result.set(MAXIMUM_KEYS.get(field), maximum);
        }
        JSONObject references = inputs.getJSONObject("referenceImages");
        int workflowMinimum = references == null ? 0 : references.getInt("minimumWhenProvided", 0);
        Integer savedMinimum = nonNegative(configured.get("minReferenceImages"), "minReferenceImages");
        if (savedMinimum == null) {
            savedMinimum = nonNegative(configured.get("minRefImages"), "minRefImages");
        }
        int configuredMinimum = savedMinimum == null ? 0 : savedMinimum;
        int minimum = Math.max(workflowMinimum, configuredMinimum);
        Integer referenceMaximum = result.getInt("maxReferenceImages");
        if (referenceMaximum != null && referenceMaximum > 0 && minimum > referenceMaximum) {
            throw invalid("参考图最少数量不能超过实际使用上限");
        }
        result.set("minReferenceImages", Boolean.TRUE.equals(result.getBool("supportReferenceImages")) ? minimum : 0);
        boolean anyImages = result.getBool("supportReferenceImages") || result.getBool("supportFirstFrame")
                || result.getBool("supportLastFrame");
        result.set("referenceImageInputFormats", anyImages ? List.of("url", "data_uri") : List.of());
        result.set("supportDataUriInput", anyImages);
        Integer imageMaximum = sumMaximum(inputs, "firstFrame", "lastFrame", "referenceImages");
        if (totalLimit != null) {
            imageMaximum = imageMaximum == null ? totalLimit : Math.min(imageMaximum, totalLimit);
        }
        result.set("maxImageInputs", imageMaximum);
        Integer savedImageMinimum = nonNegative(configured.get("minImageInputs"), "minImageInputs");
        if (savedImageMinimum == null) {
            savedImageMinimum = nonNegative(configured.get("minImages"), "minImages");
        }
        int imageMinimum = savedImageMinimum == null ? 0 : savedImageMinimum;
        if (imageMaximum != null && imageMinimum > imageMaximum) {
            throw invalid("图片输入最少数量不能超过实际使用上限");
        }
        result.set("minImageInputs", imageMinimum);
        result.set("comfyuiImageUsageMaximum", totalLimit);
        result.set("supportSeed", fields.contains("seed") && !Boolean.FALSE.equals(configured.getBool("supportSeed")));
        result.set("supportGenerateAudio", fields.contains("generateAudio")
                && !Boolean.FALSE.equals(configured.getBool("supportGenerateAudio")));
        result.set("supportWatermark", false).set("supportCameraFixed", false);
        if (!fields.contains("count")) {
            result.set("maxCount", 1);
        }
        result.set(VERSION_KEY, version.getId()).set("comfyuiWorkflowVersionNo", version.getVersionNo());
        result.set("comfyuiFields", fields).set(INPUTS_KEY, inputs);
        return result;
    }

    private Integer limit(JSONObject configured, JSONObject limits, String field) {
        if (limits != null && limits.containsKey(field)) {
            return nonNegative(limits.get(field), field);
        }
        // 既有配置没有来源标记，保留已保存的数量与禁用限制；显式 null 表示跟随工作流。
        String supportKey = SUPPORT_KEYS.get(field);
        if (supportKey != null && !configured.containsKey(supportKey)) {
            supportKey = LEGACY_SUPPORT_KEYS.get(field).stream().filter(configured::containsKey).findFirst().orElse(null);
        }
        if (supportKey != null && Boolean.FALSE.equals(configured.getBool(supportKey))) {
            return 0;
        }
        Integer maximum = nonNegative(configured.get(MAXIMUM_KEYS.get(field)), MAXIMUM_KEYS.get(field));
        if (maximum == null && "referenceImages".equals(field)) {
            maximum = nonNegative(configured.get("maxRefImages"), "maxRefImages");
        }
        if (maximum == null && "imageInputs".equals(field)) {
            maximum = nonNegative(configured.get("maxImages"), "maxImages");
        }
        if (maximum == null && "referenceVideos".equals(field)) {
            maximum = nonNegative(configured.get("maxRefVideos"), "maxRefVideos");
        }
        if (maximum == null && "referenceAudios".equals(field)) {
            maximum = nonNegative(configured.get("maxRefAudios"), "maxRefAudios");
        }
        if (maximum != null) {
            return maximum;
        }
        return null;
    }

    private void validateLimitKeys(JSONObject limits) {
        if (limits == null) {
            return;
        }
        for (String field : limits.keySet()) {
            if (!MEDIA_FIELDS.contains(field) && !"imageInputs".equals(field)) {
                throw invalid("未知的输入限制：" + field);
            }
            nonNegative(limits.get(field), field);
        }
    }

    private Integer nonNegative(Object value, String key) {
        if (value == null || JSONUtil.isNull(value)) {
            return null;
        }
        try {
            int result = new BigDecimal(value.toString()).intValueExact();
            if (result < 0) {
                throw invalid("输入限制 " + key + " 必须是非负整数");
            }
            return result;
        } catch (NumberFormatException | ArithmeticException failure) {
            throw invalid("输入限制 " + key + " 必须是非负整数");
        }
    }

    private Integer sumMaximum(JSONObject inputs, String... fields) {
        int total = 0;
        for (String field : fields) {
            JSONObject input = inputs.getJSONObject(field);
            if (input == null) {
                continue;
            }
            Integer maximum = input.getInt("maximum");
            if (maximum == null) {
                return null;
            }
            total = Math.addExact(total, maximum);
        }
        return total;
    }

    private BusinessException invalid(String message) {
        return new BusinessException(400, "ComfyUI 能力配置无效：" + message);
    }

    public static String inputLabel(String field) {
        return switch (field) {
            case "referenceImages" -> "参考图";
            case "firstFrame" -> "首帧";
            case "lastFrame" -> "尾帧";
            case "referenceVideos" -> "参考视频";
            case "referenceAudios" -> "参考音频";
            case "imageInputs" -> "图片输入总数";
            default -> field;
        };
    }
}
