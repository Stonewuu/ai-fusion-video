package com.stonewu.fusion.service.canvas.agent;

import java.util.List;

public final class CanvasAgentToolNames {
    private CanvasAgentToolNames() {
    }

    public static final List<String> NAMES = List.of("get_canvas_context", "query_canvas_nodes",
            "get_canvas_node_details", "inspect_canvas_media",
            "apply_canvas_operations", "edit_canvas_resource", "validate_canvas_run", "run_canvas_nodes",
            "get_canvas_run",
            "await_canvas_run", "cancel_canvas_run", "select_canvas_candidate", "focus_canvas_nodes",
            "create_canvas_checkpoint", "undo_canvas_command",
            "bind_canvas_output", "confirm_canvas_writeback", "arrange_canvas_nodes",
            "get_canvas_models", "get_generation_model_capabilities", "query_canvas_library",
            "get_canvas_library_resource",
            "query_canvas_outputs", "get_canvas_output", "get_canvas_fixed_output", "import_canvas_sources",
            "get_canvas_import", "await_canvas_import", "place_canvas_import",
            "control_canvas_import", "get_canvas_writeback", "review_canvas_batch", "get_canvas_mcp_operations",
            "get_canvas_mcp_call");

    public static boolean requiresCanvas(String name) {
        return name.equals("await_generation_task") || NAMES.contains(name) && !name.equals(
                "get_generation_model_capabilities");
    }
}
