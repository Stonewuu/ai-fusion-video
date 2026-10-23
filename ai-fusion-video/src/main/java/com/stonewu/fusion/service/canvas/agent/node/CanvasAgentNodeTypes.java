package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.config.ai.AiAgentRegistry;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;

import java.util.LinkedHashSet;
import java.util.Set;
import java.util.function.Function;

public final class CanvasAgentNodeTypes {
    private CanvasAgentNodeTypes() {
    }

    public static final Set<CanvasNodeKind> KINDS = Set.of(CanvasNodeKind.SCRIPT_PARSE, CanvasNodeKind.STORY_TO_SCRIPT,
            CanvasNodeKind.EPISODE_PARSE, CanvasNodeKind.STORYBOARD_GENERATE, CanvasNodeKind.ASSET_EXTRACT,
            CanvasNodeKind.ASSET_IMAGE_GENERATE, CanvasNodeKind.FRAME_GENERATE, CanvasNodeKind.AGENT);
    public static final Set<String> TYPES = Set.of("ai_media", "concept_visualizer", "story_to_script",
            "script_full_parse",
            "script_episode_parse", "script_assistant", "episode_script_creator", "episode_scene_writer",
            "script_to_storyboard",
            "episode_storyboard_writer", "storyboard_asset_preprocessor", "asset_image_gen", "asset_image_executor",
            "storyboard_frame_gen", "storyboard_frame_executor", "storyboard_video_gen", "storyboard_video_executor",
            "canvas_asset_extractor");
    public static final Set<String> CHILD_TOOLS = Set.of("episode_scene_writer", "episode_script_creator",
            "storyboard_asset_preprocessor",
            "episode_storyboard_writer", "generate_asset_image", "generate_storyboard_frame",
            "generate_storyboard_video");
    public static final Set<String> BUSINESS_TOOLS = Set.of("get_project", "get_project_script", "get_script",
            "get_script_structure", "get_script_episode",
            "get_script_scene", "get_asset", "list_project_assets", "query_asset_items", "query_asset_metadata",
            "get_project_storyboard",
            "get_storyboard", "get_storyboard_scene_items", "get_generation_model_capabilities", "get_generation_task",
            "await_generation_task",
            "update_script_info", "save_script_episode", "save_script_scene_items", "manage_script_scenes",
            "update_script_scene", "update_script",
            "delete_script_child_resource", "create_asset", "update_asset", "batch_create_assets",
            "batch_create_asset_items", "add_asset_item",
            "update_asset_item", "delete_asset_resource", "update_asset_image", "save_storyboard_episode",
            "save_storyboard_scene_shots",
            "manage_storyboard_children", "batch_update_storyboard", "insert_storyboard_item",
            "update_storyboard_scene", "update_storyboard_item",
            "delete_storyboard_child_resource", "update_storyboard_item_frame", "update_storyboard_item_video",
            "generate_image", "generate_video");

    public static boolean permittedTool(String name) {
        return BUSINESS_TOOLS.contains(name) || CHILD_TOOLS.contains(name);
    }

    public static String type(CanvasNode node) {
        String type = node.config().path("agentType").asText();
        if (!KINDS.contains(node.kind()) || !TYPES.contains(type)) {
            throw CanvasRunException.invalid("请为节点选择已登记的专业智能体");
        }
        return type;
    }

    public static Set<String> tools(AiAgentRegistry registry, AiAgentDefinition root) {
        return tools(registry::getByType, root);
    }

    public static Set<String> tools(Function<String, AiAgentDefinition> definitions, AiAgentDefinition root) {
        Set<String> visited = new LinkedHashSet<>();
        Set<String> tools = new LinkedHashSet<>();
        collect(definitions, root, visited, tools);
        return Set.copyOf(tools);
    }

    private static void collect(Function<String, AiAgentDefinition> definitions, AiAgentDefinition definition,
                                Set<String> visited, Set<String> tools) {
        if (!visited.add(definition.getType()) || !Integer.valueOf(1).equals(definition.getEnableTools())) {
            return;
        }
        if (definition.getToolNames() == null) {
            tools.addAll(BUSINESS_TOOLS);
        } else {
            tools.addAll(definition.getToolNames());
        }
        if (definition.getSubAgentTools() != null) {
            for (var child : definition.getSubAgentTools()) {
                collect(definitions, definitions.apply(child.getRefAgentType()), visited, tools);
            }
        }
    }
}
