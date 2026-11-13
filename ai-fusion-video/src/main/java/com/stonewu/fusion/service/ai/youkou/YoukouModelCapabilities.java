package com.stonewu.fusion.service.ai.youkou;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.catalog.ProviderModelCandidate.TextCapabilities;

import java.util.List;
import java.util.Map;

/** 将平台已经公布的能力转为编辑器使用的配置，不套用同名官方模型预设。 */
public final class YoukouModelCapabilities {
    private YoukouModelCapabilities() {
    }

    public static ObjectNode configuration(JsonNode entry, String endpoint) {
        ObjectNode config = YoukouContract.JSON.createObjectNode();
        var contract = config.putObject("youkouContract");
        contract.put("model", entry.path("id").asText()).put("publicModelId", entry.path("public_model_id").asText())
                .put("kind", entry.path("output_type").asText());
        contract.set("parameters", entry.path("parameters").deepCopy());
        contract.set("capabilities", entry.path("capabilities").deepCopy());
        JsonNode fields = entry.path("parameters").path(endpoint).path("fields");
        if (entry.path("output_type").asText().equals("text")) {
            config.put("apiMode", "chat_completions").put("useResponsesApi", false).put("sendIncludeReasoning", false);
            JsonNode tokens = fields.path("max_completion_tokens");
            if (tokens.path("max").isInt()) {
                config.put("maxTokens", tokens.get("max").asInt());
            }
            copyDefault(config, "reasoningEffort", fields.path("reasoning_effort"));
            return config;
        }
        config.put("supportCameraFixed", false).put("supportWatermark", false)
                .put("supportSeed", false).put("supportGenerateAudio", false);
        copyValues(config, "supportedResolutions", fields.path("resolution"));
        copyDefault(config, "defaultResolution", fields.path("resolution"));
        copyValues(config, "supportedAspectRatios", fields.path("aspect_ratio"));
        copyDefault(config, "defaultAspectRatio", fields.path("aspect_ratio"));
        JsonNode capability = entry.path("capabilities");
        JsonNode image = fields.path("input_images");
        boolean imageSupported = supported(image) && image.path("max_count").asInt() > 0;
        boolean video = entry.path("output_type").asText().equals("video");
        var modes = YoukouContract.values(capability.path("image").path("modes"));
        boolean references = imageSupported && (!video || modes.contains("reference"));
        config.put("supportReferenceImages", references)
                .put("maxReferenceImages", references ? image.path("max_count").asInt() : 0)
                .put("minReferenceImages", image.path("min_count").asInt())
                .put("supportFirstFrame", video && imageSupported && (modes.contains("first") || modes.contains("first_last")))
                .put("supportLastFrame", video && imageSupported && modes.contains("first_last"))
                .put("maxImageInputs", imageSupported ? image.path("max_count").asInt() : 0)
                .put("minImageInputs", image.path("min_count").asInt());
        config.putArray("referenceImageInputFormats").add("data_uri");
        for (String kind : List.of("video", "audio")) {
            JsonNode input = fields.path("input_" + kind + "s");
            String label = kind.equals("video") ? "Video" : "Audio";
            boolean supported = supported(input) && input.path("max_count").asInt() > 0;
            config.put("supportReference" + label + "s", supported)
                    .put("maxReference" + label + "s", supported ? input.path("max_count").asInt() : 0);
            config.putArray("reference" + label + "InputFormats").add("data_uri");
        }
        config.put("referenceAudioRequiresVisualInput",
                capability.path("constraints").path("audio_requires_image_or_video").asBoolean());
        if (video) {
            var seconds = fields.path("seconds");
            copyValues(config, "supportedDurations", seconds);
            copyDefault(config, "defaultDuration", seconds);
            if (seconds.path("values").isArray() && !seconds.path("values").isEmpty()) {
                config.put("minDuration", seconds.get("values").valueStream().mapToInt(JsonNode::asInt).min().orElseThrow());
                config.put("maxDuration", seconds.get("values").valueStream().mapToInt(JsonNode::asInt).max().orElseThrow());
            }
            var rules = config.putObject("videoRules").put("version", 1).put("profile", "youkou")
                    .put("promptRequired", fields.path("prompt").path("required").asBoolean())
                    .put("lastFrameRequiresFirst", true).put("automaticDuration", false);
            copyDefault(rules, "defaultRatio", fields.path("aspect_ratio"));
            if (fields.path("prompt").path("max_length").isInt()) {
                rules.set("promptMaxLength", fields.get("prompt").get("max_length"));
            }
            limits(rules.putObject("imageLimits"), image);
            limits(rules.putObject("referenceVideoLimits"), fields.path("input_videos"));
            limits(rules.putObject("referenceAudioLimits"), fields.path("input_audios"));
        } else if (!supported(fields.path("resolution")) || !supported(fields.path("aspect_ratio"))) {
            // 固定像素型图片模型直接展示平台允许的 size，不猜档位和比例之间的映射。
            config.put("youkouSizeOnly", true);
            copyValues(config, "supportedResolutions", fields.path("size"));
            copyDefault(config, "defaultResolution", fields.path("size"));
            config.putArray("supportedAspectRatios").add("auto");
            config.put("defaultAspectRatio", "auto");
        }
        return config;
    }

    public static String unavailable(JsonNode entry, ObjectNode config, String endpoint) {
        JsonNode fields = entry.path("parameters").path(endpoint).path("fields");
        if (!fields.isObject() || !YoukouContract.values(entry.path("parameters").path(endpoint)
                .path("content_types")).contains("application/json")) {
            return "平台尚未公布完整接口参数，请稍后重新读取";
        }
        if (entry.path("output_type").asText().equals("text")) {
            return null;
        }
        if (entry.path("output_type").asText().equals("video") && !supported(fields.path("seconds"))) {
            return "此模型未提供视频生成时长选项，暂不支持快速添加";
        }
        if (!config.path("supportedResolutions").isArray() || config.path("supportedResolutions").isEmpty()
                || !config.path("supportedAspectRatios").isArray() || config.path("supportedAspectRatios").isEmpty()) {
            return "平台尚未公布可用的输出规格，请稍后重新读取";
        }
        return null;
    }

    public static TextCapabilities text(JsonNode entry) {
        JsonNode capabilities = entry.path("capabilities");
        boolean vision = capabilities.path("features").path("vision").asText().equals("supported");
        boolean reasoning = capabilities.path("features").path("thinking").asText().equals("supported");
        JsonNode fields = entry.path("parameters").path("/v1/chat/completions").path("fields");
        Integer context = capabilities.path("effective_limits").path("context_window").isInt()
                ? capabilities.path("effective_limits").path("context_window").asInt() : null;
        return new TextCapabilities(vision ? List.of("image") : List.of(),
                vision ? Map.of("image", List.of("url", "base64")) : Map.of(), context, reasoning,
                reasoning ? YoukouContract.values(fields.path("reasoning_effort").path("values")) : List.of());
    }

    private static void copyValues(ObjectNode target, String key, JsonNode field) {
        if (supported(field) && field.path("values").isArray()) {
            target.set(key, field.get("values").deepCopy());
        }
    }

    private static void copyDefault(ObjectNode target, String key, JsonNode field) {
        if (supported(field) && field.hasNonNull("default")) {
            target.set(key, field.get("default").deepCopy());
        }
    }

    private static boolean supported(JsonNode field) {
        return field.path("status").asText().equals("supported");
    }

    private static void limits(ObjectNode target, JsonNode field) {
        var names = Map.of("min_duration_seconds", "minSeconds", "max_duration_seconds", "maxSeconds",
                "max_total_duration_seconds", "maxTotalSeconds", "max_bytes", "maxBytes",
                "min_dimension", "minDimension", "max_dimension", "maxDimension",
                "min_aspect_ratio", "minAspect", "max_aspect_ratio", "maxAspect");
        names.forEach((source, destination) -> {
            if (supported(field) && field.path(source).asDouble() > 0) {
                target.set(destination, field.get(source).deepCopy());
            }
        });
        if (supported(field) && field.path("allowed_mime_types").isArray()) {
            target.set("mimeTypes", field.get("allowed_mime_types").deepCopy());
        }
    }
}
