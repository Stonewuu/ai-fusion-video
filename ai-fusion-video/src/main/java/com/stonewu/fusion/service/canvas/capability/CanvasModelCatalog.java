package com.stonewu.fusion.service.canvas.capability;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowCapabilities;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowService;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * 实时、脱敏的模型能力目录。权限和启用状态不缓存，工作流正文沿用不可变版本缓存。
 */
@Component
@RequiredArgsConstructor
public class CanvasModelCatalog {
    public record AudioService(long id, String name) {
    }

    public record Input(Integer maximum, int minimum, int minimumWhenProvided) {
        public Input(Integer maximum, int minimum) {
            this(maximum, minimum, minimum);
        }
    }

    public record Version(long id, int number, Map<String, Input> inputs, List<String> fields,
                          Map<String, Object> parameters, String unavailable) {
    }

    public record Option(long id, String name, String kind, String resultType,
                         Map<String, Input> inputs, Map<String, Object> parameters,
                         List<Version> versions, String unavailable) {
    }

    private final AiModelService models;
    private final ApiConfigService apis;
    private final GenerationModelCapabilityService capabilities;
    private final ComfyUiWorkflowService workflows;
    private final ObjectMapper json;
    private static final Set<String> MEDIA_FIELDS = Set.of(
            "prompt", "firstFrame", "lastFrame", "referenceImages", "referenceVideos", "referenceAudios");
    private static final List<String> PARAMETERS = List.of("supportedAspectRatios", "supportedResolutions",
            "supportedSizes", "minDuration", "maxDuration", "maxDurationByResolution", "defaultDuration", "defaultResolution",
            "supportedDurations", "supportedDurationsByResolution",
            "framesWithReferenceVideos",
            "supportCameraFixed",
            "maxImageInputs", "minImageInputs", "supportGenerateAudio", "supportWatermark", "supportSeed", "videoRules",
            "promptExpansionMode", "referenceTaskType", "outputFormat", "promptExtend");

    public List<Option> list() {
        return models.getEnabledList().stream().map(this::describe).toList();
    }

    public List<AudioService> audioServices() {
        return apis.getListByPlatform("sonilo").stream().map(api -> new AudioService(api.getId(), api.getName()))
                .toList();
    }

    public void requireAudio(long id) {
        var api = apis.getById(id);
        if (api == null || !Integer.valueOf(1).equals(api.getStatus()) || !"sonilo".equals(api.getPlatform())) {
            throw new BusinessException(400, "请选择已启用的 Sonilo 音频服务");
        }
    }

    public Option require(long id) {
        AiModel model = models.getById(id);
        if (model == null || !Integer.valueOf(1).equals(model.getStatus())) {
            throw new BusinessException(400, "所选模型已停用或不存在，请重新选择");
        }
        Option option = describe(model);
        if (option.unavailable() != null) {
            throw new BusinessException(400, option.unavailable());
        }
        return option;
    }

    private Option describe(AiModel model) {
        var api = model.getApiConfigId() == null ? null : apis.getById(model.getApiConfigId());
        String result = switch (model.getModelType()) {
            case 1 -> "text";
            case 2 -> "image";
            case 3 -> "video";
            default -> "unsupported";
        };
        boolean comfy = "comfyui".equals(model.getModelProtocol()) || (api != null && "comfyui".equals(
                api.getPlatform()));
        String kind = comfy ? "comfyui" : result;
        if (api == null || !Integer.valueOf(1).equals(api.getStatus())) {
            return new Option(model.getId(), model.getName(), kind, result, Map.of(), Map.of(), List.of(),
                    "模型的供应商配置已停用或不存在");
        }
        if ("unsupported".equals(result)) {
            return new Option(model.getId(), model.getName(), kind, result, Map.of(), Map.of(), List.of(),
                    "此模型类型不支持画布生成");
        }
        Map<String, Input> inputs = new LinkedHashMap<>();
        inputs.put("prompt", new Input(1, 0));
        Map<String, Object> parameters = new LinkedHashMap<>();
        List<Version> versions = new ArrayList<>();
        try {
            if (comfy) {
                workflows.validateModelBinding(model, api);
                for (var version : workflows.getVersions(model.getComfyuiWorkflowId())) {
                    if (!Boolean.TRUE.equals(version.getPublished())) {
                        continue;
                    }
                    try {
                        var resolved = capabilities.getMergedModelConfig(model, version.getId());
                        var resolvedInputs = resolved.getJSONObject(ComfyUiWorkflowCapabilities.INPUTS_KEY);
                        Map<String, Input> versionInputs = new LinkedHashMap<>();
                        var fields = resolved.getJSONArray("comfyuiFields").toList(String.class);
                        for (String field : fields) {
                            if (!MEDIA_FIELDS.contains(field)) {
                                continue;
                            }
                            var input = resolvedInputs.getJSONObject(field);
                            versionInputs.put(field, new Input(input.getInt("maximum"), input.getInt("minimum", 0),
                                    input.getInt("minimumWhenProvided", 0)));
                        }
                        if (versionInputs.containsKey("prompt")) {
                            versionInputs.put("context", new Input(1000, 0));
                        }
                        versions.add(
                                new Version(version.getId(), version.getVersionNo(), Map.copyOf(versionInputs), fields,
                                        parameters(resolved), null));
                    } catch (BusinessException error) {
                        versions.add(new Version(version.getId(), version.getVersionNo(), Map.of(), List.of(),
                                Map.of(), error.getMessage()));
                    }
                }
                inputs.clear();
                if (versions.isEmpty()) {
                    throw new BusinessException("工作流没有可用的已发布版本");
                }
            } else if ("text".equals(result)) {
                if (model.getMultimodalInputTypes() != null && model.getMultimodalInputTypes().contains("image")) {
                    inputs.put("images", new Input(8, 0));
                }
                parameters.put("multimodalInputTypes",
                        model.getMultimodalInputTypes() == null ? List.of() : model.getMultimodalInputTypes());
                parameters.put("multimodalInputTransports",
                        model.getMultimodalInputTransports() == null ? Map.of() : model.getMultimodalInputTransports());
            } else {
                JSONObject snapshot = "image".equals(result)
                        ? capabilities.buildImageCapabilitySnapshot(model) : capabilities.buildVideoCapabilitySnapshot(
                        model);
                for (String key : PARAMETERS) {
                    if (snapshot.get(key) != null) {
                        parameters.put(key, json.convertValue(snapshot.get(key), Object.class));
                    }
                }
                if ("image".equals(result)) {
                    var image = capabilities.resolveImageCapability(model);
                    if (image.supportsReferenceImages()) {
                        inputs.put("referenceImages",
                                new Input(image.maxReferenceImages(), image.minReferenceImages()));
                    }
                } else {
                    var video = capabilities.resolveVideoCapability(model);
                    if (video.supportsFirstFrame()) {
                        inputs.put("firstFrame", new Input(1, 0));
                    }
                    if (video.supportsLastFrame()) {
                        inputs.put("lastFrame", new Input(1, 0));
                    }
                    if (video.supportsReferenceImages()) {
                        inputs.put("referenceImages", new Input(video.maxReferenceImages(), 0));
                    }
                    if (video.supportsReferenceVideos()) {
                        inputs.put("referenceVideos", new Input(video.maxReferenceVideos(), 0));
                    }
                    if (video.supportsReferenceAudios()) {
                        inputs.put("referenceAudios", new Input(video.maxReferenceAudios(), 0));
                    }
                    JSONObject config = capabilities.getMergedModelConfig(model);
                    parameters.put("exclusiveInputModes",
                            enabled(config, "exclusiveInputModes", "mutuallyExclusiveInputModes"));
                    parameters.put("referenceAudioRequiresVisualInput",
                            enabled(config, "referenceAudioRequiresVisualInput", "audioRequiresVisualInput"));
                }
            }
        } catch (BusinessException | IllegalArgumentException error) {
            return new Option(model.getId(), model.getName(), kind, result, Map.of(), Map.of(), List.of(),
                    "模型或工作流能力配置无效，请在系统设置中检查");
        }
        if (inputs.containsKey("prompt")) {
            inputs.put("context", new Input(1000, 0));
        }
        return new Option(model.getId(), model.getName(), kind, result, Map.copyOf(inputs), Map.copyOf(parameters),
                List.copyOf(versions), null);
    }

    private static boolean enabled(JSONObject config, String first, String second) {
        return Boolean.TRUE.equals(config.getBool(config.containsKey(first) ? first : second));
    }

    private Map<String, Object> parameters(JSONObject config) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (String key : PARAMETERS) {
            if (config.get(key) != null && !JSONUtil.isNull(config.get(key))) {
                result.put(key, json.convertValue(config.get(key), Object.class));
            }
        }
        result.put("exclusiveInputModes", enabled(config, "exclusiveInputModes", "mutuallyExclusiveInputModes"));
        result.put("referenceAudioRequiresVisualInput", enabled(config, "referenceAudioRequiresVisualInput", "audioRequiresVisualInput"));
        return Map.copyOf(result);
    }

    public static Option forVersion(Option model, long versionId) {
        var version = model.versions().stream().filter(item -> item.id() == versionId).findFirst()
                .orElseThrow(() -> new BusinessException("请选择此模型绑定工作流的已发布版本"));
        if (version.unavailable() != null) {
            throw new BusinessException(400, version.unavailable());
        }
        return new Option(model.id(), model.name(), model.kind(), model.resultType(), version.inputs(),
                version.parameters(), model.versions(), null);
    }
}
