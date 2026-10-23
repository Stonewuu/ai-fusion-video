package com.stonewu.fusion.service.ai.tool.storyboard;

import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Set;

@Component
@RequiredArgsConstructor
public class ManageStoryboardChildrenToolExecutor implements ToolExecutor {
    private final StoryboardToolCommands tools;

    @Override
    public String getToolName() {
        return "manage_storyboard_children";
    }

    @Override
    public String getDisplayName() {
        return "编辑分镜集合";
    }

    @Override
    public String getToolDescription() {
        return "按父集合的实际版本追加、重建、清空、删除或排序分镜集、场次、镜头。数组顺序是业务顺序；必须使用真实 ID。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "resourceType": {
                            "type": "string",
                            "enum": [
                                "storyboard",
                                "storyboardEpisode",
                                "storyboardScene"
                            ]
                        },
                        "resourceId": {
                            "type": "integer",
                            "minimum": 1
                        },
                        "expectedVersion": {
                            "type": "integer",
                            "minimum": 0
                        },
                        "mode": {
                            "type": "string",
                            "enum": [
                                "APPEND",
                                "REPLACE",
                                "CLEAR",
                                "DELETE",
                                "REORDER"
                            ]
                        },
                        "children": {
                            "type": "array",
                            "maxItems": 500,
                            "items": {
                                "type": "object"
                            },
                            "description": "APPEND/REPLACE 的新建对象；分集可带 scenes、场次可带 shots。其它模式传空数组。"
                        },
                        "childIds": {
                            "type": "array",
                            "maxItems": 5000,
                            "items": {
                                "type": "integer",
                                "minimum": 1
                            },
                            "description": "DELETE 的对象或 REORDER 的完整顺序；其它模式传空数组。"
                        }
                    },
                    "required": [
                        "resourceType",
                        "resourceId",
                        "expectedVersion",
                        "mode",
                        "children",
                        "childIds"
                    ],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public boolean mayRequireHighRiskApproval() {
        return true;
    }

    @Override
    public ToolPermissionRisk getPermissionRisk(Map<String, Object> input) {
        return Set.of("REPLACE", "CLEAR", "DELETE")
                .contains(String.valueOf(input.get("mode"))) ? ToolPermissionRisk.HIGH_RISK : ToolPermissionRisk.EDIT;
    }

    @Override
    public String execute(String input, ToolExecutionContext context) {
        return tools.execute(getToolName(), input, context);
    }
}
