package com.stonewu.fusion.service.canvas.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Arrays;

public enum CanvasNodeKind {
    PROJECT("project"), STORY("story"), TEXT("text"), SCRIPT("script"),
    SCRIPT_EPISODE("scriptEpisode"), SCRIPT_SCENE("scriptScene"), STORYBOARD("storyboard"),
    STORYBOARD_EPISODE("storyboardEpisode"), STORYBOARD_SCENE("storyboardScene"), SHOT("shot"),
    ASSET("asset"), ASSET_ITEM("assetItem"), IMAGE("image"), VIDEO("video"), AUDIO("audio"),
    TEXT_FILE("textFile"), DOCUMENT("document"),
    TEXT_GENERATION("textGeneration"), TEXT_COMBINE("textCombine"),
    SCRIPT_PARSE("scriptParse"), STORY_TO_SCRIPT("storyToScript"), EPISODE_PARSE("episodeParse"),
    STORYBOARD_GENERATE("storyboardGenerate"), ASSET_EXTRACT("assetExtract"),
    ASSET_IMAGE_GENERATE("assetImageGenerate"), FRAME_GENERATE("frameGenerate"),
    IMAGE_GENERATION("imageGeneration"), VIDEO_GENERATION("videoGeneration"),
    COMFYUI("comfyui"), FRAME_EXTRACT("frameExtract"), IMAGE_TRANSFORM("imageTransform"),
    MUSIC_GENERATION("musicGeneration"), SFX_GENERATION("sfxGeneration"),
    AUDIO_MIX("audioMix"), VIDEO_COMPOSE("videoCompose"), EXPORT("export"),
    AGENT("agent"), MCP_TOOL("mcpTool"), SELECTION("selection"), COLLECTION("collection"), FOREACH("foreach"),
    WRITEBACK("writeback"), GROUP("group"), NOTE("note");

    private final String code;

    CanvasNodeKind(String code) {
        this.code = code;
    }

    @JsonValue
    public String code() {
        return code;
    }

    public boolean storedMedia() {
        return switch (this) {
            case IMAGE, VIDEO, AUDIO, TEXT_FILE, DOCUMENT -> true;
            default -> false;
        };
    }

    @JsonCreator
    public static CanvasNodeKind fromCode(String value) {
        return Arrays.stream(values()).filter(kind -> kind.code.equals(value)).findFirst()
                .orElseThrow(() -> new CanvasException(CanvasErrorCode.UNKNOWN_NODE_KIND,
                        "不支持的画布节点类型: " + value));
    }
}
