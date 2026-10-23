package com.stonewu.fusion.service.canvas.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Arrays;

public enum CanvasValueType {
    TEXT("text", false), IMAGE("image", false), VIDEO("video", false), AUDIO("audio", false), DOCUMENT("document",
            false),
    PROJECT("project", true), SCRIPT("script", true), SCRIPT_EPISODE("scriptEpisode", true),
    SCRIPT_SCENE("scriptScene", true), STORYBOARD("storyboard", true),
    STORYBOARD_EPISODE("storyboardEpisode", true), STORYBOARD_SCENE("storyboardScene", true),
    SHOT("shot", true), ASSET("asset", true), ASSET_ITEM("assetItem", true);

    private final String code;
    private final boolean businessResource;

    CanvasValueType(String code, boolean businessResource) {
        this.code = code;
        this.businessResource = businessResource;
    }

    @JsonValue
    public String code() {
        return code;
    }

    public boolean isBusinessResource() {
        return businessResource;
    }

    @JsonCreator
    public static CanvasValueType fromCode(String value) {
        return Arrays.stream(values()).filter(type -> type.code.equals(value)).findFirst()
                .orElseThrow(() -> new CanvasException(CanvasErrorCode.INVALID_PORT,
                        "不支持的画布数据类型: " + value));
    }
}
