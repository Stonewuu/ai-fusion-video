package com.stonewu.fusion.service.ai.configuration;

import com.stonewu.fusion.service.canvas.agent.CanvasAgentToolNames;

/**
 * 设置目录的展示分类；不改变工具注册、白名单或执行权限。
 */
public final class AgentToolCategories {
    private AgentToolCategories() {
    }

    public static String category(String name) {
        if (CanvasAgentToolNames.NAMES.contains(name) && !name.equals("get_generation_model_capabilities")) {
            return "画布操作";
        }
        return switch (name) {
            case "get_project", "list_my_projects", "save_project", "delete_project" -> "项目管理";
            case "get_project_script", "get_script", "get_script_structure", "get_script_episode", "get_script_scene",
                 "update_script_info", "save_script_episode", "save_script_scene_items", "manage_script_scenes",
                 "update_script_scene", "update_script", "delete_script_child_resource" -> "剧本创作";
            case "get_project_storyboard", "get_storyboard", "get_storyboard_scene_items", "save_storyboard_episode",
                 "save_storyboard_scene_shots", "manage_storyboard_children", "batch_update_storyboard",
                 "insert_storyboard_item",
                 "update_storyboard_scene", "update_storyboard_item", "delete_storyboard_child_resource",
                 "update_storyboard_item_frame", "update_storyboard_item_video" -> "分镜创作";
            case "get_asset", "list_project_assets", "query_asset_items", "query_asset_metadata", "create_asset",
                 "update_asset",
                 "batch_create_assets", "batch_create_asset_items", "add_asset_item", "update_asset_item",
                 "delete_asset_resource",
                 "update_asset_image" -> "资产管理";
            case "get_generation_model_capabilities", "generate_image", "generate_video", "get_generation_task",
                 "await_generation_task", "delete_image_task", "delete_video_task" -> "媒体生成";
            default -> throw new IllegalStateException("请为平台工具登记展示分类：" + name);
        };
    }
}
