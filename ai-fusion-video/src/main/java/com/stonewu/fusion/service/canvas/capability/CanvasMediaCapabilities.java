package com.stonewu.fusion.service.canvas.capability;

import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.capability.CanvasConfigField.*;
import static com.stonewu.fusion.service.canvas.capability.CanvasDefinitions.*;

final class CanvasMediaCapabilities {
    private CanvasMediaCapabilities() {
    }

    static List<CanvasNodeCapability> definitions() {
        List<CanvasNodeCapability> result = new ArrayList<>();
        result.add(media(CanvasNodeKind.IMAGE, "图片", CanvasValueType.IMAGE));
        result.add(media(CanvasNodeKind.VIDEO, "视频", CanvasValueType.VIDEO));
        result.add(media(CanvasNodeKind.AUDIO, "音频", CanvasValueType.AUDIO));
        result.add(media(CanvasNodeKind.TEXT_FILE, "文本文件", CanvasValueType.TEXT));
        result.add(media(CanvasNodeKind.DOCUMENT, "PDF 文档", CanvasValueType.DOCUMENT));
        result.add(capability(CanvasNodeKind.IMAGE_GENERATION, "生成图片", "generation", "image", false,
                fields(field("modelId", integer(1, MAX_SAFE_ID, false)), field("prompt", text(32000)),
                        field("ratio", text(32)), field("resolution", text(64)), field("count", integer(1, 64, false)),
                        field("width", integer(1, 16384, false)), field("height", integer(1, 16384, false))),
                List.of(port("prompt", CanvasValueType.TEXT, false, true), contextPort(),
                        port("referenceImages", CanvasValueType.IMAGE, true, false)),
                List.of(port("images", CanvasValueType.IMAGE, true, false))));
        result.add(capability(CanvasNodeKind.VIDEO_GENERATION, "生成视频", "generation", "video", false,
                videoFields(), videoInputs(), List.of(port("videos", CanvasValueType.VIDEO, true, false),
                        port("firstFrames", CanvasValueType.IMAGE, true, false),
                        port("lastFrames", CanvasValueType.IMAGE, true, false))));
        result.add(capability(CanvasNodeKind.COMFYUI, "ComfyUI 工作流", "generation", "comfyui", false,
                fields(field("modelId", integer(1, MAX_SAFE_ID, false)),
                        field("workflowVersionId", integer(1, MAX_SAFE_ID, false)),
                        field("resultType", choice(true, "image", "video")), field("prompt", text(32000)),
                        field("negativePrompt", text(32000)), field("fps", integer(1, 120, false)),
                        field("generateAudio", flag()),
                        field("width", integer(1, 16384, false)), field("height", integer(1, 16384, false)),
                        field("count", integer(1, 64, false)), field("duration", integer(1, 3600, false)),
                        field("seed", integer(-1, MAX_SAFE_ID, false))),
                videoInputs(), List.of(typedPort("results", "resultType", true, false))));
        result.add(capability(CanvasNodeKind.FRAME_EXTRACT, "提取首尾帧", "processing", "media", false,
                fields(field("frameType", choice(true, "first", "last", "both"))),
                List.of(port("video", CanvasValueType.VIDEO, false, true)),
                List.of(port("firstFrame", CanvasValueType.IMAGE, false, false),
                        port("lastFrame", CanvasValueType.IMAGE, false, false))));
        result.add(capability(CanvasNodeKind.IMAGE_TRANSFORM, "裁剪与调整尺寸", "processing", "media", false,
                fields(field("operation", choice(true, "crop", "resize")), field("x", integer(0, 100000, false)),
                        field("y", integer(0, 100000, false)), field("width", integer(1, 16384, false)),
                        field("height", integer(1, 16384, false)),
                        field("fit", choice(false, "contain", "cover", "fill"))),
                List.of(port("image", CanvasValueType.IMAGE, false, true)),
                List.of(port("image", CanvasValueType.IMAGE, false, false))));
        result.add(audio(CanvasNodeKind.MUSIC_GENERATION, "生成配乐", "music"));
        result.add(audio(CanvasNodeKind.SFX_GENERATION, "生成音效", "sfx"));
        result.add(capability(CanvasNodeKind.AUDIO_MIX, "混音", "delivery", "compose", false,
                fields(field("musicVolume", number(0, 1)), field("originalVolume", number(0, 1)),
                        field("sfxVolume", number(0, 1))),
                List.of(port("video", CanvasValueType.VIDEO, false, true),
                        port("music", CanvasValueType.AUDIO, false, false),
                        port("sfx", CanvasValueType.AUDIO, false, false)),
                List.of(port("video", CanvasValueType.VIDEO, false, false))));
        result.add(capability(CanvasNodeKind.VIDEO_COMPOSE, "合成视频", "delivery", "compose", false,
                fields(field("resolution", text(64)), field("frameRate", integer(1, 120, false))),
                List.of(port("videos", CanvasValueType.VIDEO, true, false),
                        port("episode", CanvasValueType.STORYBOARD_EPISODE, false, false)),
                List.of(port("video", CanvasValueType.VIDEO, false, false))));
        return result;
    }

    private static CanvasNodeCapability media(CanvasNodeKind kind, String label, CanvasValueType type) {
        return capability(kind, label, "media", "resource", false,
                fields(field("resourceId", text(36))), List.of(), List.of(port("media", type, false, false)));
    }

    private static CanvasNodeCapability audio(CanvasNodeKind kind, String label, String operation) {
        return capability(kind, label, "generation", "audio", false,
                fields(field("apiConfigId", integer(1, MAX_SAFE_ID, false)), field("prompt", text(32000)),
                        field("operation", choice(true, operation))),
                List.of(port("video", CanvasValueType.VIDEO, false, true)),
                List.of(port("audio", CanvasValueType.AUDIO, false, false)));
    }

    private static Map<String, CanvasConfigField> videoFields() {
        return fields(field("modelId", integer(1, MAX_SAFE_ID, false)), field("prompt", text(32000)),
                field("ratio", text(32)), field("resolution", text(64)), field("duration", integer(-1, 3600, false)),
                field("count", integer(1, 64, false)), field("seed", integer(-1, MAX_SAFE_ID, false)),
                field("watermark", flag()), field("generateAudio", flag()), field("cameraFixed", flag()));
    }

    private static List<CanvasPortDefinition> videoInputs() {
        return List.of(port("prompt", CanvasValueType.TEXT, false, true), contextPort(),
                port("firstFrame", CanvasValueType.IMAGE, false, false),
                port("lastFrame", CanvasValueType.IMAGE, false, false),
                port("referenceImages", CanvasValueType.IMAGE, true, false),
                port("referenceVideos", CanvasValueType.VIDEO, true, false),
                port("referenceAudios", CanvasValueType.AUDIO, true, false));
    }
}
