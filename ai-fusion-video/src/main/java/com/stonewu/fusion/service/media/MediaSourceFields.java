package com.stonewu.fusion.service.media;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;

import java.util.Map;

final class MediaSourceFields {
    private MediaSourceFields() {
    }

    record Value(String uri, MediaKind kind) {
    }

    static Map<String, MediaKind> fields(CanvasValueType type) {
        return switch (type) {
            case PROJECT -> Map.of("coverUrl", MediaKind.IMAGE, "artStyleImageUrl", MediaKind.IMAGE);
            case ASSET -> Map.of("coverUrl", MediaKind.IMAGE, "properties.videoUrl", MediaKind.VIDEO,
                    "properties.audioUrl", MediaKind.AUDIO, "properties.thumbnailUrl", MediaKind.IMAGE,
                    "properties.sourceMediaUrl", MediaKind.IMAGE);
            case ASSET_ITEM -> Map.of("imageUrl", MediaKind.IMAGE, "thumbnailUrl", MediaKind.IMAGE,
                    "properties.videoUrl", MediaKind.VIDEO, "properties.audioUrl", MediaKind.AUDIO,
                    "properties.thumbnailUrl", MediaKind.IMAGE, "properties.sourceMediaUrl", MediaKind.IMAGE);
            case SHOT -> Map.of("imageUrl", MediaKind.IMAGE, "referenceImageUrl", MediaKind.IMAGE,
                    "generatedImageUrl", MediaKind.IMAGE, "firstFrameImageUrl", MediaKind.IMAGE,
                    "lastFrameImageUrl", MediaKind.IMAGE, "videoUrl", MediaKind.VIDEO,
                    "generatedVideoUrl", MediaKind.VIDEO, "customData.videoPosterUrl", MediaKind.IMAGE);
            case STORYBOARD_EPISODE -> Map.of("composedVideoUrl", MediaKind.VIDEO,
                    "bgmAudioUrl", MediaKind.AUDIO, "bgmSfxAudioUrl", MediaKind.AUDIO);
            default -> throw MediaException.invalid("此业务类型没有登记媒体来源");
        };
    }

    static Value get(CanvasValueType type, ObjectNode data, String field) {
        if (field == null || field.isBlank()) {
            throw MediaException.invalid("必须明确指定媒体来源字段");
        }
        MediaKind kind = fields(type).get(field);
        if (field.equals(
                "properties.sourceMediaUrl") && (type == CanvasValueType.ASSET || type == CanvasValueType.ASSET_ITEM)) {
            kind = switch (data.path("properties").path("sourceGenerationMode").asText("")) {
                case "image" -> MediaKind.IMAGE;
                case "video" -> MediaKind.VIDEO;
                case "audio" -> MediaKind.AUDIO;
                default -> throw MediaException.invalid("来源素材缺少明确的 sourceGenerationMode，不能猜测类型");
            };
        }
        if (kind == null) {
            throw MediaException.invalid("此字段没有登记媒体来源映射");
        }
        JsonNode value = data;
        for (String part : field.split("\\.")) {
            value = value.path(part);
        }
        if (!value.isTextual() || value.textValue().isBlank()) {
            throw new MediaException(422, "MEDIA_SOURCE_EMPTY", "指定的业务字段尚无媒体内容");
        }
        return new Value(value.textValue(), kind);
    }
}
