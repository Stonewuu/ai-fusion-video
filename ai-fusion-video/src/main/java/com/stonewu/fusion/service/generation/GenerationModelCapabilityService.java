package com.stonewu.fusion.service.generation;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.ModelPresetService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * 生成模型能力解析与输入校验。
 * <p>
 * 目标：在真正发起平台请求前，就能根据当前模型能力给出稳定、可解释的错误，避免静默忽略或将错误延后到平台侧。
 */
@Service
@RequiredArgsConstructor
public class GenerationModelCapabilityService {

    private final AiModelMetadataResolver aiModelMetadataResolver;
    private final ModelPresetService modelPresetService;

    public ImageModelCapability resolveImageCapability(AiModel model) {
        return resolveImageCapability(model, aiModelMetadataResolver.resolvePlatform(model));
    }

    public ImageModelCapability resolveImageCapability(AiModel model, String platform) {
        JSONObject config = getMergedModelConfig(model);

        Boolean supportsReferenceImages = getBoolean(config,
                "supportReferenceImages", "supportsReferenceImages", "supportRefImages", "supportImageReferences");
        Integer minReferenceImages = getInteger(config, "minReferenceImages", "minRefImages");
        Integer maxReferenceImages = getInteger(config, "maxReferenceImages", "maxRefImages");

        boolean finalSupportsReferenceImages = supportsReferenceImages != null
                ? supportsReferenceImages : false;
        int finalMinReferenceImages = minReferenceImages != null
                ? Math.max(minReferenceImages, 0) : 0;
        Integer finalMaxReferenceImages = maxReferenceImages != null ? Math.max(maxReferenceImages, 0) : null;

        if (!finalSupportsReferenceImages) {
            finalMinReferenceImages = 0;
            finalMaxReferenceImages = 0;
        }

        return new ImageModelCapability(finalSupportsReferenceImages, finalMinReferenceImages, finalMaxReferenceImages);
    }

    public VideoModelCapability resolveVideoCapability(AiModel model) {
        return resolveVideoCapability(model, aiModelMetadataResolver.resolvePlatform(model));
    }

    public VideoModelCapability resolveVideoCapability(AiModel model, String platform) {
        JSONObject config = getMergedModelConfig(model);

        Boolean supportsFirstFrame = getBoolean(config,
                "supportFirstFrame", "supportsFirstFrame", "allowFirstFrame", "supportStartImage");
        Boolean supportsLastFrame = getBoolean(config,
                "supportLastFrame", "supportsLastFrame", "allowLastFrame", "supportEndImage");
        Boolean supportsReferenceImages = getBoolean(config,
                "supportReferenceImages", "supportsReferenceImages", "supportRefImages", "supportImageReferences");
        Boolean supportsReferenceVideos = getBoolean(config,
                "supportReferenceVideos", "supportsReferenceVideos", "supportRefVideos", "supportVideoReferences");
        Boolean supportsReferenceAudios = getBoolean(config,
                "supportReferenceAudios", "supportsReferenceAudios", "supportRefAudios", "supportAudioReferences");

        Integer minImageInputs = getInteger(config, "minImageInputs", "minImages");
        Integer maxImageInputs = getInteger(config, "maxImageInputs", "maxImages");
        Integer maxReferenceImages = getInteger(config, "maxReferenceImages", "maxRefImages");
        Integer maxReferenceVideos = getInteger(config, "maxReferenceVideos", "maxRefVideos");
        Integer maxReferenceAudios = getInteger(config, "maxReferenceAudios", "maxRefAudios");

        boolean finalSupportsFirstFrame = supportsFirstFrame != null
                ? supportsFirstFrame : false;
        boolean finalSupportsLastFrame = supportsLastFrame != null
                ? supportsLastFrame : false;
        boolean finalSupportsReferenceImages = supportsReferenceImages != null
                ? supportsReferenceImages : false;
        boolean finalSupportsReferenceVideos = supportsReferenceVideos != null
                ? supportsReferenceVideos : false;
        boolean finalSupportsReferenceAudios = supportsReferenceAudios != null
                ? supportsReferenceAudios : false;

        int finalMinImageInputs = minImageInputs != null ? Math.max(minImageInputs, 0) : 0;
        Integer finalMaxImageInputs = maxImageInputs != null ? Math.max(maxImageInputs, 0) : null;
        Integer finalMaxReferenceImages = maxReferenceImages != null ? Math.max(maxReferenceImages, 0) : null;
        Integer finalMaxReferenceVideos = maxReferenceVideos != null ? Math.max(maxReferenceVideos, 0) : null;
        Integer finalMaxReferenceAudios = maxReferenceAudios != null ? Math.max(maxReferenceAudios, 0) : null;

        if (!finalSupportsReferenceImages) {
            finalMaxReferenceImages = 0;
        }
        if (!finalSupportsReferenceVideos) {
            finalMaxReferenceVideos = 0;
        }
        if (!finalSupportsReferenceAudios) {
            finalMaxReferenceAudios = 0;
        }

        return new VideoModelCapability(
                finalSupportsFirstFrame,
                finalSupportsLastFrame,
                finalSupportsReferenceImages,
                finalSupportsReferenceVideos,
                finalSupportsReferenceAudios,
                finalMinImageInputs,
                finalMaxImageInputs,
                finalMaxReferenceImages,
                finalMaxReferenceVideos,
                finalMaxReferenceAudios
        );
    }

    public void validateImageTask(AiModel model, ImageTask task) {
        validateImageTask(model, task, aiModelMetadataResolver.resolvePlatform(model));
    }

    public void validateImageTask(AiModel model, ImageTask task, String platform) {
        if (model == null || task == null) {
            return;
        }
        ImageModelCapability capability = resolveImageCapability(model, platform);
        List<String> referenceImages = parseJsonUrls(task.getRefImageUrls());

        if (!referenceImages.isEmpty() && !capability.supportsReferenceImages()) {
            throw new BusinessException("当前图片模型 " + modelLabel(model)
                    + " 不支持参考图输入，请不要传 imageUrls；如需图生图，请切换到支持参考图的模型。");
        }

        if (capability.maxReferenceImages() != null && referenceImages.size() > capability.maxReferenceImages()) {
            throw new BusinessException("当前图片模型 " + modelLabel(model)
                    + " 最多支持 " + capability.maxReferenceImages() + " 张参考图，当前传入了 " + referenceImages.size() + " 张。");
        }

        if (capability.minReferenceImages() > 0 && !referenceImages.isEmpty()
                && referenceImages.size() < capability.minReferenceImages()) {
            throw new BusinessException("当前图片模型 " + modelLabel(model)
                    + " 至少需要 " + capability.minReferenceImages() + " 张参考图，当前仅传入了 " + referenceImages.size() + " 张。");
        }
    }

    public void validateVideoTask(AiModel model, VideoTask task) {
        validateVideoTask(model, task, aiModelMetadataResolver.resolvePlatform(model));
    }

    public void validateVideoTask(AiModel model, VideoTask task, String platform) {
        if (model == null || task == null) {
            return;
        }
        JSONObject config = getMergedModelConfig(model);
        VideoModelCapability capability = resolveVideoCapability(model, platform);
        List<String> referenceImages = parseJsonUrls(task.getReferenceImageUrls());
        List<String> referenceVideos = parseJsonUrls(task.getReferenceVideoUrls());
        List<String> referenceAudios = parseJsonUrls(task.getReferenceAudioUrls());
        boolean hasFirstFrame = StrUtil.isNotBlank(task.getFirstFrameImageUrl());
        boolean hasLastFrame = StrUtil.isNotBlank(task.getLastFrameImageUrl());
        int totalImageInputs = referenceImages.size() + (hasFirstFrame ? 1 : 0) + (hasLastFrame ? 1 : 0);

        boolean exclusiveInputModes = Boolean.TRUE.equals(getBoolean(config,
                "exclusiveInputModes", "mutuallyExclusiveInputModes"));
        boolean hasFrameInputs = hasFirstFrame || hasLastFrame;
        boolean hasReferenceMedia = !referenceImages.isEmpty()
                || !referenceVideos.isEmpty() || !referenceAudios.isEmpty();
        if (exclusiveInputModes && hasFrameInputs && hasReferenceMedia) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 的首帧/首尾帧模式与多模态参考模式互斥，请只选择一种输入方式。");
        }

        boolean audioRequiresVisualInput = Boolean.TRUE.equals(getBoolean(config,
                "referenceAudioRequiresVisualInput", "audioRequiresVisualInput"));
        if (audioRequiresVisualInput && !referenceAudios.isEmpty()
                && !hasFrameInputs && referenceImages.isEmpty() && referenceVideos.isEmpty()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 不支持仅输入参考音频，请同时传入参考图片或参考视频。");
        }

        if (hasFirstFrame && !capability.supportsFirstFrame()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 不支持首帧图输入，请不要传 firstFrameImageUrl。");
        }
        if (hasLastFrame && !capability.supportsLastFrame()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 不支持尾帧图输入，请不要传 lastFrameImageUrl。");
        }
        if (!referenceImages.isEmpty() && !capability.supportsReferenceImages()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 不支持 referenceImageUrls，请改用支持多图参考的模型。");
        }
        if (!referenceVideos.isEmpty() && !capability.supportsReferenceVideos()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 不支持 referenceVideoUrls。");
        }
        if (!referenceAudios.isEmpty() && !capability.supportsReferenceAudios()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 不支持 referenceAudioUrls。");
        }

        if (capability.minImageInputs() > 0 && totalImageInputs < capability.minImageInputs()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 至少需要 " + capability.minImageInputs() + " 张图片输入，请传 firstFrameImageUrl、lastFrameImageUrl 或 referenceImageUrls。");
        }
        if (capability.maxImageInputs() != null && totalImageInputs > capability.maxImageInputs()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 最多支持 " + capability.maxImageInputs() + " 张图片输入，当前传入了 " + totalImageInputs + " 张。");
        }
        if (capability.maxReferenceImages() != null && referenceImages.size() > capability.maxReferenceImages()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 最多支持 " + capability.maxReferenceImages() + " 张 referenceImageUrls，当前传入了 " + referenceImages.size() + " 张。");
        }
        if (capability.maxReferenceVideos() != null && referenceVideos.size() > capability.maxReferenceVideos()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 最多支持 " + capability.maxReferenceVideos() + " 个 referenceVideoUrls，当前传入了 " + referenceVideos.size() + " 个。");
        }
        if (capability.maxReferenceAudios() != null && referenceAudios.size() > capability.maxReferenceAudios()) {
            throw new BusinessException("当前视频模型 " + modelLabel(model)
                    + " 最多支持 " + capability.maxReferenceAudios() + " 个 referenceAudioUrls，当前传入了 " + referenceAudios.size() + " 个。");
        }
    }

    public String describeImageCapability(AiModel model) {
        if (model == null) {
            return "当前未配置默认图片模型。";
        }
        ImageModelCapability capability = resolveImageCapability(model);
        if (!capability.supportsReferenceImages()) {
            return "当前默认图片模型：" + modelLabel(model) + "；不支持参考图，仅支持文生图。";
        }
        String limitText = capability.maxReferenceImages() != null && capability.maxReferenceImages() > 0
                ? "，最多 " + capability.maxReferenceImages() + " 张" : "";
        return "当前默认图片模型：" + modelLabel(model) + "；支持参考图" + limitText + "。";
    }

    public JSONObject buildImageCapabilitySnapshot(AiModel model) {
        if (model == null) {
            return JSONUtil.createObj()
                    .set("configured", false)
                    .set("summary", "当前未配置默认图片模型。");
        }

        AiModelMetadata metadata = aiModelMetadataResolver.resolve(model);
        String platform = metadata.platform();
        JSONObject config = getMergedModelConfig(model);
        ImageModelCapability capability = resolveImageCapability(model, platform);

        return JSONUtil.createObj()
                .set("configured", true)
                .set("modelType", "image")
                .set("modelId", model.getId())
                .set("modelName", modelLabel(model))
                .set("modelCode", model.getCode())
                .set("capabilityPresetCode", model.getCapabilityPresetCode())
                .set("modelProtocol", metadata.modelProtocol())
                .set("platform", platform)
                .set("supportsReferenceImages", capability.supportsReferenceImages())
                .set("minReferenceImages", capability.minReferenceImages())
                .set("maxReferenceImages", capability.maxReferenceImages())
                .set("asyncMode", Boolean.TRUE.equals(getBoolean(config,
                    "asyncMode", "useAsyncMode", "asyncTaskMode", "enableAsyncTask", "asyncEnabled")))
                .set("supportedAspectRatios", getStringList(config, "supportedAspectRatios"))
                .set("supportedSizes", copyJsonObject(config, "supportedSizes"))
                .set("defaultWidth", getInteger(config, "defaultWidth"))
                .set("defaultHeight", getInteger(config, "defaultHeight"))
                .set("summary", describeImageCapability(model));
    }

    public String describeVideoCapability(AiModel model) {
        if (model == null) {
            return "当前未配置默认视频模型。";
        }

        VideoModelCapability capability = resolveVideoCapability(model);
        List<String> parts = new ArrayList<>();
        parts.add("当前默认视频模型：" + modelLabel(model));
        parts.add("首帧图：" + yesNo(capability.supportsFirstFrame()));
        parts.add("尾帧图：" + yesNo(capability.supportsLastFrame()));
        parts.add("参考图：" + referenceSupportText(capability.supportsReferenceImages(), capability.maxReferenceImages(), "张"));
        parts.add("参考视频：" + referenceSupportText(capability.supportsReferenceVideos(), capability.maxReferenceVideos(), "个"));
        parts.add("参考音频：" + referenceSupportText(capability.supportsReferenceAudios(), capability.maxReferenceAudios(), "个"));
        if (capability.minImageInputs() > 0) {
            parts.add("至少需要 " + capability.minImageInputs() + " 张图片输入");
        }
        return String.join("；", parts) + "。";
    }

    public JSONObject buildVideoCapabilitySnapshot(AiModel model) {
        if (model == null) {
            return JSONUtil.createObj()
                    .set("configured", false)
                    .set("summary", "当前未配置默认视频模型。");
        }

        AiModelMetadata metadata = aiModelMetadataResolver.resolve(model);
        String platform = metadata.platform();
        JSONObject config = getMergedModelConfig(model);
        VideoModelCapability capability = resolveVideoCapability(model, platform);

        return JSONUtil.createObj()
                .set("configured", true)
                .set("modelType", "video")
                .set("modelId", model.getId())
                .set("modelName", modelLabel(model))
                .set("modelCode", model.getCode())
                .set("capabilityPresetCode", model.getCapabilityPresetCode())
                .set("modelProtocol", metadata.modelProtocol())
                .set("platform", platform)
                .set("supportsFirstFrame", capability.supportsFirstFrame())
                .set("supportsLastFrame", capability.supportsLastFrame())
                .set("supportsReferenceImages", capability.supportsReferenceImages())
                .set("supportsReferenceVideos", capability.supportsReferenceVideos())
                .set("supportsReferenceAudios", capability.supportsReferenceAudios())
                .set("minImageInputs", capability.minImageInputs())
                .set("maxImageInputs", capability.maxImageInputs())
                .set("maxReferenceImages", capability.maxReferenceImages())
                .set("maxReferenceVideos", capability.maxReferenceVideos())
                .set("maxReferenceAudios", capability.maxReferenceAudios())
                .set("supportedAspectRatios", getStringList(config, "supportedAspectRatios"))
                .set("supportedResolutions", getStringList(config, "supportedResolutions"))
                .set("minDuration", getInteger(config, "minDuration"))
                .set("maxDuration", getInteger(config, "maxDuration"))
                .set("defaultDuration", getInteger(config, "defaultDuration"))
                .set("supportCameraFixed", getBoolean(config, "supportCameraFixed"))
                .set("summary", describeVideoCapability(model));
    }

    public JSONObject getMergedModelConfig(AiModel model) {
        JSONObject merged = new JSONObject();
        if (model != null && modelPresetService != null && StrUtil.isNotBlank(model.getCapabilityPresetCode())) {
            mergeConfig(merged, parseConfig(modelPresetService.getPresetConfig(model.getCapabilityPresetCode())));
        }
        if (model != null) {
            mergeConfig(merged, parseConfig(model.getConfig()));
        }
        return merged;
    }

    public String resolveModelPlatform(AiModel model) {
        return aiModelMetadataResolver.resolvePlatform(model);
    }

    private JSONObject parseConfig(String configJson) {
        if (StrUtil.isBlank(configJson)) {
            return new JSONObject();
        }
        try {
            return JSONUtil.parseObj(configJson);
        } catch (Exception ignored) {
            return new JSONObject();
        }
    }

    private void mergeConfig(JSONObject target, JSONObject source) {
        if (target == null || source == null || source.isEmpty()) {
            return;
        }
        for (String key : source.keySet()) {
            target.set(key, source.get(key));
        }
    }

    private String modelLabel(AiModel model) {
        if (model == null) {
            return "未命名模型";
        }
        return StrUtil.blankToDefault(model.getName(), model.getCode());
    }

    private Boolean getBoolean(JSONObject config, String... keys) {
        for (String key : keys) {
            if (!config.containsKey(key)) {
                continue;
            }
            Object value = config.get(key);
            if (value instanceof Boolean bool) {
                return bool;
            }
            if (value != null) {
                String text = value.toString().trim();
                if ("true".equalsIgnoreCase(text) || "1".equals(text) || "yes".equalsIgnoreCase(text)) {
                    return true;
                }
                if ("false".equalsIgnoreCase(text) || "0".equals(text) || "no".equalsIgnoreCase(text)) {
                    return false;
                }
            }
        }
        return null;
    }

    private Integer getInteger(JSONObject config, String... keys) {
        for (String key : keys) {
            if (!config.containsKey(key)) {
                continue;
            }
            try {
                return config.getInt(key);
            } catch (Exception ignored) {
                Object value = config.get(key);
                if (value != null) {
                    try {
                        return Integer.parseInt(value.toString());
                    } catch (NumberFormatException ignoredAgain) {
                        return null;
                    }
                }
            }
        }
        return null;
    }

    private List<String> getStringList(JSONObject config, String key) {
        if (config == null || StrUtil.isBlank(key) || !config.containsKey(key)) {
            return List.of();
        }
        Object value = config.get(key);
        if (value instanceof JSONArray array) {
            return array.toList(String.class).stream()
                    .filter(StrUtil::isNotBlank)
                    .map(String::trim)
                    .toList();
        }
        if (value instanceof List<?> list) {
            return list.stream()
                    .map(item -> item == null ? null : item.toString())
                    .filter(StrUtil::isNotBlank)
                    .map(String::trim)
                    .toList();
        }
        if (value != null) {
            String text = value.toString().trim();
            return StrUtil.isBlank(text) ? List.of() : List.of(text);
        }
        return List.of();
    }

    private JSONObject copyJsonObject(JSONObject config, String key) {
        if (config == null || StrUtil.isBlank(key) || !config.containsKey(key)) {
            return null;
        }
        Object value = config.get(key);
        if (value instanceof JSONObject jsonObject) {
            return JSONUtil.parseObj(jsonObject);
        }
        if (value != null) {
            try {
                return JSONUtil.parseObj(value);
            } catch (Exception ignored) {
                return null;
            }
        }
        return null;
    }

    private List<String> parseJsonUrls(String json) {
        if (StrUtil.isBlank(json)) {
            return List.of();
        }
        try {
            JSONArray array = JSONUtil.parseArray(json);
            return array.toList(String.class).stream()
                    .filter(StrUtil::isNotBlank)
                    .map(String::trim)
                    .toList();
        } catch (Exception ignored) {
            return List.of();
        }
    }

    private String yesNo(boolean value) {
        return value ? "支持" : "不支持";
    }

    private String referenceSupportText(boolean supported, Integer maxCount, String unit) {
        if (!supported) {
            return "不支持";
        }
        if (maxCount != null && maxCount > 0) {
            return "支持，最多 " + maxCount + unit;
        }
        return "支持";
    }

    public record ImageModelCapability(boolean supportsReferenceImages,
                                       int minReferenceImages,
                                       Integer maxReferenceImages) {
    }

    public record VideoModelCapability(boolean supportsFirstFrame,
                                       boolean supportsLastFrame,
                                       boolean supportsReferenceImages,
                                       boolean supportsReferenceVideos,
                                       boolean supportsReferenceAudios,
                                       int minImageInputs,
                                       Integer maxImageInputs,
                                       Integer maxReferenceImages,
                                       Integer maxReferenceVideos,
                                       Integer maxReferenceAudios) {
    }
}
