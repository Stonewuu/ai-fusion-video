package com.stonewu.fusion.service.generation.job;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;

import java.util.ArrayList;
import java.util.List;

/** 只公开本次生成的创作输入，不返回内部执行上下文或供应商鉴权配置。 */
public record GenerationExecutionInput(String prompt, ObjectNode parameters, List<Reference> references) {
    private static final List<String> PARAMETERS = List.of("generateMode", "count", "ratio", "resolution",
            "width", "height", "duration", "generateAudio", "cameraFixed", "watermark", "seed", "workflowVersionId");

    public record Reference(String kind, String source) {
    }

    public static GenerationExecutionInput from(ObjectNode input, int count, CanvasJson json) {
        ObjectNode parameters = input.objectNode();
        for (String field : PARAMETERS) {
            if (input.has(field)) {
                parameters.set(field, input.get(field).deepCopy());
            }
        }
        parameters.put("count", count);
        List<Reference> references = new ArrayList<>();
        one(references, input, "firstFrameImageUrl", "firstFrame");
        one(references, input, "lastFrameImageUrl", "lastFrame");
        many(references, input, "refImageUrls", "image", json);
        many(references, input, "referenceImageUrls", "image", json);
        many(references, input, "referenceVideoUrls", "video", json);
        many(references, input, "referenceAudioUrls", "audio", json);
        return new GenerationExecutionInput(input.path("prompt").textValue(), parameters, List.copyOf(references));
    }

    private static void one(List<Reference> references, ObjectNode input, String field, String kind) {
        JsonNode value = input.path(field);
        if (!value.isMissingNode() && !value.isNull() && !value.asText().isBlank()) {
            references.add(new Reference(kind, value.asText()));
        }
    }

    private static void many(List<Reference> references, ObjectNode input, String field, String kind, CanvasJson json) {
        JsonNode value = input.path(field);
        if (!value.isMissingNode() && !value.isNull() && !value.asText().isBlank()) {
            for (String source : json.read(value.asText(), String[].class)) {
                references.add(new Reference(kind, source));
            }
        }
    }
}
