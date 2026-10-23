package com.stonewu.fusion.repository.resource;

import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

/** 业务资源类型与存储表的固定对应，用于元数据查询及边界校验。 */
public final class ResourceTables {
    private ResourceTables() {
    }

    public static String of(CanvasValueType type) {
        return switch (type) {
            case PROJECT -> "afv_project";
            case SCRIPT -> "afv_script";
            case SCRIPT_EPISODE -> "afv_script_episode";
            case SCRIPT_SCENE -> "afv_script_scene_item";
            case STORYBOARD -> "afv_storyboard";
            case STORYBOARD_EPISODE -> "afv_storyboard_episode";
            case STORYBOARD_SCENE -> "afv_storyboard_scene";
            case SHOT -> "afv_storyboard_item";
            case ASSET -> "afv_asset";
            case ASSET_ITEM -> "afv_asset_item";
            default -> throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "不是可读取的业务对象类型");
        };
    }
}
