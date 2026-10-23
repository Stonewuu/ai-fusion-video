package com.stonewu.fusion.service.media.model;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.Arrays;

public enum MediaSourceType {
    PROJECT("project", CanvasValueType.PROJECT),
    ASSET("asset", CanvasValueType.ASSET),
    ASSET_ITEM("assetItem", CanvasValueType.ASSET_ITEM),
    SHOT("shot", CanvasValueType.SHOT),
    STORYBOARD_EPISODE("storyboardEpisode", CanvasValueType.STORYBOARD_EPISODE),
    IMAGE_ITEM("imageItem", null),
    VIDEO_ITEM("videoItem", null);

    private final String code;
    private final CanvasValueType businessType;

    MediaSourceType(String code, CanvasValueType businessType) {
        this.code = code;
        this.businessType = businessType;
    }

    @JsonValue
    public String code() {
        return code;
    }

    public boolean generation() {
        return this == IMAGE_ITEM || this == VIDEO_ITEM;
    }

    public CanvasValueType businessType() {
        if (businessType == null) {
            throw MediaException.invalid("生成条目不是可编辑业务对象");
        }
        return businessType;
    }

    public static MediaSourceType business(CanvasValueType type) {
        return Arrays.stream(values()).filter(value -> value.businessType == type && type != null)
                .findFirst().orElseThrow(() -> MediaException.invalid("业务对象没有媒体来源定义"));
    }

    @JsonCreator
    public static MediaSourceType fromCode(String code) {
        return Arrays.stream(values()).filter(value -> value.code.equals(code))
                .findFirst().orElseThrow(() -> MediaException.invalid("媒体来源类型无效"));
    }
}
