package com.stonewu.fusion.service.ai.googleflow;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.enums.ai.AiModelTypeEnum;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Google Flow Reverse API 的模型解析与结果提取工具。
 */
public final class GoogleFlowReverseApiSupport {

    public static final String PLATFORM = "GoogleFlowReverseApi";
    public static final String DEFAULT_BASE_URL = "http://localhost:8000";

    private static final Pattern MARKDOWN_IMAGE_PATTERN = Pattern.compile("!\\[[^\\]]*]\\((.*?)\\)");
    private static final Pattern HTML_VIDEO_PATTERN = Pattern.compile("<video[^>]+src=['\"](.*?)['\"]", Pattern.CASE_INSENSITIVE);
    private static final Pattern SIZE_PATTERN = Pattern.compile("(\\d{3,5})\\s*[xX]\\s*(\\d{3,5})");

    private static final Set<String> IMAGE_ALIAS_MODELS = Set.of(
            "gemini-2.5-flash-image",
            "gemini-3.0-pro-image",
            "gemini-3.1-flash-image",
            "imagen-4.0-generate-preview"
    );

    private static final Map<String, String> IMAGE_ASPECT_SUFFIXES = Map.ofEntries(
            Map.entry("16:9", "landscape"),
            Map.entry("9:16", "portrait"),
            Map.entry("1:1", "square"),
            Map.entry("4:3", "four-three"),
            Map.entry("3:4", "three-four")
    );

    private static final Map<String, Map<String, String>> VIDEO_BASE_MODELS = Map.ofEntries(
            Map.entry("veo_3_1_t2v_fast", Map.of("landscape", "veo_3_1_t2v_fast_landscape", "portrait", "veo_3_1_t2v_fast_portrait")),
            Map.entry("veo_3_1_t2v_fast_ultra", Map.of("landscape", "veo_3_1_t2v_fast_ultra_landscape", "portrait", "veo_3_1_t2v_fast_portrait_ultra")),
            Map.entry("veo_3_1_t2v_fast_ultra_relaxed", Map.of("landscape", "veo_3_1_t2v_fast_ultra_relaxed_landscape", "portrait", "veo_3_1_t2v_fast_portrait_ultra_relaxed")),
            Map.entry("veo_3_1_t2v", Map.of("landscape", "veo_3_1_t2v_landscape", "portrait", "veo_3_1_t2v_portrait")),
            Map.entry("veo_3_1_t2v_lite", Map.of("landscape", "veo_3_1_t2v_lite_landscape", "portrait", "veo_3_1_t2v_lite_portrait")),
            Map.entry("veo_3_1_i2v_s_fast_fl", Map.of("landscape", "veo_3_1_i2v_s_fast_fl", "portrait", "veo_3_1_i2v_s_fast_portrait_fl")),
            Map.entry("veo_3_1_i2v_s_fast_ultra_fl", Map.of("landscape", "veo_3_1_i2v_s_fast_ultra_fl", "portrait", "veo_3_1_i2v_s_fast_portrait_ultra_fl")),
            Map.entry("veo_3_1_i2v_s_fast_ultra_relaxed", Map.of("landscape", "veo_3_1_i2v_s_fast_ultra_relaxed", "portrait", "veo_3_1_i2v_s_fast_portrait_ultra_relaxed")),
            Map.entry("veo_3_1_i2v_s", Map.of("landscape", "veo_3_1_i2v_s", "portrait", "veo_3_1_i2v_s_portrait")),
            Map.entry("veo_3_1_i2v_lite", Map.of("landscape", "veo_3_1_i2v_lite", "portrait", "veo_3_1_i2v_lite_portrait")),
            Map.entry("veo_3_1_interpolation_lite", Map.of("landscape", "veo_3_1_interpolation_lite", "portrait", "veo_3_1_interpolation_lite_portrait")),
            Map.entry("veo_3_1_r2v_fast", Map.of("landscape", "veo_3_1_r2v_fast", "portrait", "veo_3_1_r2v_fast_portrait")),
            Map.entry("veo_3_1_r2v_fast_ultra", Map.of("landscape", "veo_3_1_r2v_fast_ultra", "portrait", "veo_3_1_r2v_fast_portrait_ultra")),
            Map.entry("veo_3_1_r2v_fast_ultra_relaxed", Map.of("landscape", "veo_3_1_r2v_fast_ultra_relaxed", "portrait", "veo_3_1_r2v_fast_portrait_ultra_relaxed"))
    );

    private static final Map<String, Map<String, Map<String, String>>> VIDEO_UPSAMPLE_MODELS = Map.ofEntries(
            Map.entry("veo_3_1_t2v_fast", Map.of(
                    "1080p", Map.of("landscape", "veo_3_1_t2v_fast_1080p", "portrait", "veo_3_1_t2v_fast_portrait_1080p"),
                    "4k", Map.of("landscape", "veo_3_1_t2v_fast_4k", "portrait", "veo_3_1_t2v_fast_portrait_4k")
            )),
            Map.entry("veo_3_1_t2v_fast_ultra", Map.of(
                    "1080p", Map.of("landscape", "veo_3_1_t2v_fast_ultra_1080p", "portrait", "veo_3_1_t2v_fast_portrait_ultra_1080p"),
                    "4k", Map.of("landscape", "veo_3_1_t2v_fast_ultra_4k", "portrait", "veo_3_1_t2v_fast_portrait_ultra_4k")
            )),
            Map.entry("veo_3_1_i2v_s_fast_ultra_fl", Map.of(
                    "1080p", Map.of("landscape", "veo_3_1_i2v_s_fast_ultra_fl_1080p", "portrait", "veo_3_1_i2v_s_fast_portrait_ultra_fl_1080p"),
                    "4k", Map.of("landscape", "veo_3_1_i2v_s_fast_ultra_fl_4k", "portrait", "veo_3_1_i2v_s_fast_portrait_ultra_fl_4k")
            )),
            Map.entry("veo_3_1_r2v_fast_ultra", Map.of(
                    "1080p", Map.of("landscape", "veo_3_1_r2v_fast_ultra_1080p", "portrait", "veo_3_1_r2v_fast_portrait_ultra_1080p"),
                    "4k", Map.of("landscape", "veo_3_1_r2v_fast_ultra_4k", "portrait", "veo_3_1_r2v_fast_portrait_ultra_4k")
            ))
    );

    private GoogleFlowReverseApiSupport() {
    }

    public static Integer inferRemoteModelType(String modelCode) {
        if (StrUtil.isBlank(modelCode)) {
            return null;
        }
        String normalizedCode = modelCode.trim();
        if (IMAGE_ALIAS_MODELS.contains(normalizedCode)
                || normalizedCode.contains("-image")
                || normalizedCode.startsWith("imagen-")) {
            return AiModelTypeEnum.IMAGE.getType();
        }
        if (VIDEO_BASE_MODELS.containsKey(normalizedCode) || normalizedCode.startsWith("veo_")) {
            return AiModelTypeEnum.VIDEO.getType();
        }
        return null;
    }

    public static String normalizeBaseUrl(String baseUrl) {
        if (StrUtil.isBlank(baseUrl)) {
            return baseUrl;
        }
        String normalized = baseUrl.trim().replaceAll("/+$", "");
        return normalized.replaceAll("(?i)/v1$", "");
    }

    public static String resolveBaseUrl(String baseUrl) {
        String normalizedBaseUrl = normalizeBaseUrl(baseUrl);
        if (StrUtil.isBlank(normalizedBaseUrl)) {
            return DEFAULT_BASE_URL;
        }
        return normalizedBaseUrl;
    }

    public static String requireBaseUrl(ApiConfig apiConfig) {
        return resolveBaseUrl(apiConfig != null ? apiConfig.getApiUrl() : null);
    }

    public static JSONObject parseModelConfig(AiModel model) {
        if (model == null || StrUtil.isBlank(model.getConfig())) {
            return new JSONObject();
        }
        try {
            return JSONUtil.parseObj(model.getConfig());
        } catch (Exception ignored) {
            return new JSONObject();
        }
    }

    public static ResolvedImageRequest resolveImageRequest(AiModel model, ImageTask task) {
        String modelCode = model != null ? model.getCode() : null;
        if (StrUtil.isBlank(modelCode)) {
            throw new BusinessException("未配置 GoogleFlowReverseApi 图片模型");
        }

        JSONObject config = parseModelConfig(model);
        String aspectRatio = resolveImageAspectRatio(task, config);
        String resolution = resolveImageResolution(task, config);
        validateImageSelection(config, aspectRatio, resolution, modelCode);

        return new ResolvedImageRequest(resolveImageModelCode(modelCode, aspectRatio, resolution), aspectRatio, resolution);
    }

    public static String resolveImageModelCode(String modelCode, String aspectRatio, String resolution) {
        if (StrUtil.isBlank(modelCode) || !IMAGE_ALIAS_MODELS.contains(modelCode)) {
            return modelCode;
        }
        String suffix = IMAGE_ASPECT_SUFFIXES.get(aspectRatio);
        if (suffix == null) {
            throw new BusinessException("GoogleFlowReverseApi 图片比例不支持: " + aspectRatio);
        }
        if ("1K".equalsIgnoreCase(resolution)) {
            return modelCode + "-" + suffix;
        }
        return modelCode + "-" + suffix + "-" + resolution.toLowerCase(Locale.ROOT);
    }

    public static String deriveImageAspectRatio(Integer width, Integer height) {
        return deriveNearestAspectRatio(width, height, List.of("16:9", "9:16", "1:1", "4:3", "3:4"));
    }

    public static String deriveImageResolution(Integer width, Integer height) {
        if (width == null || height == null || width <= 0 || height <= 0) {
            return "1K";
        }
        int max = Math.max(width, height);
        if (max <= 1792) {
            return "1K";
        }
        if (max <= 2688) {
            return "2K";
        }
        return "4K";
    }

    public static ResolvedVideoRequest resolveVideoRequest(AiModel model, VideoTask task) {
        String modelCode = model != null ? model.getCode() : null;
        if (StrUtil.isBlank(modelCode)) {
            throw new BusinessException("未配置 GoogleFlowReverseApi 视频模型");
        }

        JSONObject config = parseModelConfig(model);
        String aspectRatio = resolveVideoAspectRatio(task, config);
        String resolution = resolveVideoResolution(task, config);
        int imageCount = countVideoInputImages(task);

        validateVideoSelection(config, aspectRatio, resolution, imageCount, modelCode);

        return new ResolvedVideoRequest(resolveVideoModelCode(modelCode, aspectRatio, resolution), aspectRatio, resolution, imageCount);
    }

    public static String resolveVideoModelCode(String modelCode, String aspectRatio, String resolution) {
        if (!VIDEO_BASE_MODELS.containsKey(modelCode)) {
            if (StrUtil.isNotBlank(resolution) && !modelCode.toLowerCase(Locale.ROOT).contains(resolution.toLowerCase(Locale.ROOT))) {
                throw new BusinessException("模型 " + modelCode + " 不支持通过分辨率参数切换，请改用 GoogleFlowReverseApi 别名模型");
            }
            return modelCode;
        }

        String orientation = "9:16".equals(aspectRatio) ? "portrait" : "landscape";
        if (StrUtil.isBlank(resolution)) {
            return VIDEO_BASE_MODELS.get(modelCode).get(orientation);
        }

        Map<String, Map<String, String>> resolutionMap = VIDEO_UPSAMPLE_MODELS.get(modelCode);
        if (resolutionMap == null) {
            throw new BusinessException("模型 " + modelCode + " 不支持 " + resolution + " 分辨率");
        }

        Map<String, String> orientationMap = resolutionMap.get(resolution.toLowerCase(Locale.ROOT));
        if (orientationMap == null || !orientationMap.containsKey(orientation)) {
            throw new BusinessException("模型 " + modelCode + " 不支持 " + resolution + " / " + aspectRatio + " 组合");
        }
        return orientationMap.get(orientation);
    }

    public static int countVideoInputImages(VideoTask task) {
        int count = 0;
        if (task == null) {
            return count;
        }
        if (StrUtil.isNotBlank(task.getFirstFrameImageUrl())) {
            count++;
        }
        if (StrUtil.isNotBlank(task.getLastFrameImageUrl())) {
            count++;
        }
        count += parseJsonUrls(task.getReferenceImageUrls()).size();
        return count;
    }

    public static List<String> parseJsonUrls(String json) {
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

    public static String extractImageUrl(String content) {
        return extractMediaUrl(content, MARKDOWN_IMAGE_PATTERN);
    }

    public static String extractVideoUrl(String content) {
        return extractMediaUrl(content, HTML_VIDEO_PATTERN);
    }

    private static String extractMediaUrl(String content, Pattern pattern) {
        if (StrUtil.isBlank(content)) {
            return null;
        }
        Matcher matcher = pattern.matcher(content);
        if (matcher.find()) {
            return matcher.group(1).trim();
        }
        String trimmed = content.trim();
        if (trimmed.startsWith("http://") || trimmed.startsWith("https://") || trimmed.startsWith("/media/")) {
            return trimmed;
        }
        return null;
    }

    private static String resolveImageAspectRatio(ImageTask task, JSONObject config) {
        String explicit = normalizeImageAspectRatio(firstNonBlank(task != null ? task.getAspectRatio() : null, task != null ? task.getRatio() : null));
        if (StrUtil.isNotBlank(explicit)) {
            return explicit;
        }
        String derived = deriveNearestAspectRatio(
                task != null ? task.getWidth() : null,
                task != null ? task.getHeight() : null,
                getStringList(config, "supportedAspectRatios")
        );
        if (StrUtil.isNotBlank(derived)) {
            return derived;
        }
        List<String> supported = getStringList(config, "supportedAspectRatios");
        if (supported.contains("16:9")) {
            return "16:9";
        }
        if (!supported.isEmpty()) {
            return supported.getFirst();
        }
        return "16:9";
    }

    private static String resolveImageResolution(ImageTask task, JSONObject config) {
        String explicit = normalizeImageResolution(firstNonBlank(task != null ? task.getResolution() : null, null));
        if (StrUtil.isNotBlank(explicit)) {
            return explicit;
        }
        String derived = deriveImageResolution(task != null ? task.getWidth() : null, task != null ? task.getHeight() : null);
        if (StrUtil.isNotBlank(derived)) {
            return derived;
        }
        List<String> supported = getSupportedImageSizes(config);
        if (!supported.isEmpty()) {
            return supported.stream().min(Comparator.comparingInt(GoogleFlowReverseApiSupport::imageResolutionOrder)).orElse("1K");
        }
        return "1K";
    }

    private static void validateImageSelection(JSONObject config, String aspectRatio, String resolution, String modelCode) {
        List<String> supportedAspectRatios = getStringList(config, "supportedAspectRatios");
        if (!supportedAspectRatios.isEmpty() && !supportedAspectRatios.contains(aspectRatio)) {
            throw new BusinessException("模型 " + modelCode + " 不支持比例 " + aspectRatio);
        }

        List<String> supportedSizes = getSupportedImageSizes(config);
        if (!supportedSizes.isEmpty() && !supportedSizes.contains(resolution)) {
            throw new BusinessException("模型 " + modelCode + " 不支持尺寸档位 " + resolution);
        }

        JSONObject sizeMap = config.getJSONObject("supportedSizes");
        if (sizeMap != null && sizeMap.containsKey(resolution)) {
            JSONObject resolutionMap = sizeMap.getJSONObject(resolution);
            if (resolutionMap != null && !resolutionMap.isEmpty() && !resolutionMap.containsKey(aspectRatio)) {
                throw new BusinessException("模型 " + modelCode + " 不支持 " + resolution + " / " + aspectRatio + " 组合");
            }
        }
    }

    private static String resolveVideoAspectRatio(VideoTask task, JSONObject config) {
        String explicit = normalizeVideoAspectRatio(task != null ? task.getRatio() : null);
        if (StrUtil.isNotBlank(explicit)) {
            return explicit;
        }
        List<String> supported = getStringList(config, "supportedAspectRatios");
        if (supported.contains("16:9")) {
            return "16:9";
        }
        if (!supported.isEmpty()) {
            return supported.getFirst();
        }
        return "16:9";
    }

    private static String resolveVideoResolution(VideoTask task, JSONObject config) {
        String explicit = normalizeVideoResolution(task != null ? task.getResolution() : null);
        if (StrUtil.isNotBlank(explicit)) {
            return explicit;
        }
        return null;
    }

    private static void validateVideoSelection(JSONObject config, String aspectRatio, String resolution,
                                               int imageCount, String modelCode) {
        List<String> supportedAspectRatios = getStringList(config, "supportedAspectRatios");
        if (!supportedAspectRatios.isEmpty() && !supportedAspectRatios.contains(aspectRatio)) {
            throw new BusinessException("模型 " + modelCode + " 不支持比例 " + aspectRatio);
        }

        List<String> supportedResolutions = getStringList(config, "supportedResolutions");
        if (StrUtil.isNotBlank(resolution) && !supportedResolutions.isEmpty() && !supportedResolutions.contains(resolution)) {
            throw new BusinessException("模型 " + modelCode + " 不支持分辨率 " + resolution);
        }

        Integer minImages = getInteger(config, "minImageInputs", "minImages");
        Integer maxImages = getInteger(config, "maxImageInputs", "maxImages");
        if (minImages != null && imageCount < minImages) {
            throw new BusinessException("模型 " + modelCode + " 至少需要 " + minImages + " 张参考图");
        }
        if (maxImages != null && imageCount > maxImages) {
            throw new BusinessException("模型 " + modelCode + " 最多支持 " + maxImages + " 张参考图");
        }
    }

    private static List<String> getSupportedImageSizes(JSONObject config) {
        JSONObject supportedSizes = config.getJSONObject("supportedSizes");
        if (supportedSizes == null || supportedSizes.isEmpty()) {
            return List.of();
        }
        return supportedSizes.keySet().stream()
                .map(GoogleFlowReverseApiSupport::normalizeImageResolution)
                .filter(StrUtil::isNotBlank)
                .distinct()
                .sorted(Comparator.comparingInt(GoogleFlowReverseApiSupport::imageResolutionOrder))
                .toList();
    }

    private static Integer getInteger(JSONObject config, String... keys) {
        for (String key : keys) {
            if (config == null || !config.containsKey(key)) {
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

    private static List<String> getStringList(JSONObject config, String key) {
        JSONArray array = config.getJSONArray(key);
        if (array == null) {
            return List.of();
        }
        List<String> result = new ArrayList<>();
        for (Object value : array) {
            if (value == null) {
                continue;
            }
            String normalized = switch (key) {
                case "supportedAspectRatios" -> normalizeImageAspectRatio(value.toString());
                case "supportedResolutions" -> normalizeVideoResolution(value.toString());
                default -> value.toString().trim();
            };
            if (StrUtil.isNotBlank(normalized) && !result.contains(normalized)) {
                result.add(normalized);
            }
        }
        return result;
    }

    private static String normalizeImageAspectRatio(String rawValue) {
        if (StrUtil.isBlank(rawValue)) {
            return null;
        }
        String value = rawValue.trim().toLowerCase(Locale.ROOT).replace('_', '-');
        return switch (value) {
            case "16:9", "16/9", "landscape" -> "16:9";
            case "9:16", "9/16", "portrait" -> "9:16";
            case "1:1", "1/1", "square" -> "1:1";
            case "4:3", "4/3", "four-three" -> "4:3";
            case "3:4", "3/4", "three-four" -> "3:4";
            default -> value.matches("\\d+:\\d+") ? value.toUpperCase(Locale.ROOT) : null;
        };
    }

    private static String normalizeVideoAspectRatio(String rawValue) {
        String normalized = normalizeImageAspectRatio(rawValue);
        if ("16:9".equals(normalized) || "9:16".equals(normalized)) {
            return normalized;
        }
        return null;
    }

    private static String normalizeImageResolution(String rawValue) {
        if (StrUtil.isBlank(rawValue)) {
            return null;
        }
        String value = rawValue.trim().toLowerCase(Locale.ROOT).replace(" ", "");
        Matcher matcher = SIZE_PATTERN.matcher(value);
        if (matcher.matches()) {
            return deriveImageResolution(Integer.parseInt(matcher.group(1)), Integer.parseInt(matcher.group(2)));
        }
        return switch (value) {
            case "1k", "1024", "default", "standard", "base", "low" -> "1K";
            case "2k", "2048", "medium" -> "2K";
            case "4k", "4096", "high", "ultra" -> "4K";
            default -> null;
        };
    }

    private static String normalizeVideoResolution(String rawValue) {
        if (StrUtil.isBlank(rawValue)) {
            return null;
        }
        String value = rawValue.trim().toLowerCase(Locale.ROOT).replace(" ", "");
        Matcher matcher = SIZE_PATTERN.matcher(value);
        if (matcher.matches()) {
            int width = Integer.parseInt(matcher.group(1));
            int height = Integer.parseInt(matcher.group(2));
            int max = Math.max(width, height);
            if (max >= 3800) {
                return "4K";
            }
            if (max >= 1900) {
                return "1080P";
            }
            return null;
        }
        return switch (value) {
            case "default", "standard", "base", "auto", "720p", "720", "默认" -> null;
            case "1080p", "1080", "fhd", "fullhd" -> "1080P";
            case "4k", "2160p", "2160" -> "4K";
            default -> null;
        };
    }

    private static String deriveNearestAspectRatio(Integer width, Integer height, List<String> supportedRatios) {
        if (width == null || height == null || width <= 0 || height <= 0) {
            return null;
        }
        double actualRatio = width.doubleValue() / height.doubleValue();
        List<String> candidates = supportedRatios != null && !supportedRatios.isEmpty()
                ? supportedRatios.stream().filter(Objects::nonNull).toList()
                : List.of("16:9", "9:16", "1:1", "4:3", "3:4");

        return candidates.stream()
                .min(Comparator.comparingDouble(candidate -> Math.abs(parseAspectRatio(candidate) - actualRatio)))
                .orElse(null);
    }

    private static double parseAspectRatio(String ratio) {
        if (StrUtil.isBlank(ratio) || !ratio.contains(":")) {
            return 1D;
        }
        String[] parts = ratio.split(":");
        if (parts.length != 2) {
            return 1D;
        }
        try {
            return Double.parseDouble(parts[0]) / Double.parseDouble(parts[1]);
        } catch (NumberFormatException ignored) {
            return 1D;
        }
    }

    private static int imageResolutionOrder(String resolution) {
        return switch (resolution) {
            case "1K" -> 1;
            case "2K" -> 2;
            case "4K" -> 4;
            default -> 99;
        };
    }

    private static String firstNonBlank(String first, String second) {
        if (StrUtil.isNotBlank(first)) {
            return first;
        }
        if (StrUtil.isNotBlank(second)) {
            return second;
        }
        return null;
    }

    public record ResolvedImageRequest(String actualModelCode, String aspectRatio, String resolution) {
    }

    public record ResolvedVideoRequest(String actualModelCode, String aspectRatio, String resolution, int imageCount) {
    }
}