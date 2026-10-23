package com.stonewu.fusion.service.ai.tool.script;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 删除剧本模块资源工具。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DeleteScriptChildResourceToolExecutor implements ToolExecutor {

    private final ScriptToolCommands commands;

    private static final String RESOURCE_EPISODE = "episode";
    private static final String RESOURCE_SCENE = "scene";


    @Override
    public String getToolName() {
        return "delete_script_child_resource";
    }

    @Override
    public String getDisplayName() {
        return "删除剧本子资源";
    }

    @Override
    public String getToolDescription() {
        return "删除剧本分集或场次。顶层剧本由项目生命周期维护，不允许删除。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "expectedVersion": { "type": "integer", "minimum": 0, "description": "刚刚读取的对象版本；创建分集传剧本版本，更新分集传分集版本" },
                        "resourceType": {
                            "type": "string",
                            "enum": ["episode", "scene"],
                            "description": "要删除的资源类型"
                        },
                        "resourceId": {"type": "integer", "description": "要删除的资源ID"}
                    },
                    "required": ["expectedVersion", "resourceType", "resourceId"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        return commands.delete(toolInput, context);
    }
}
