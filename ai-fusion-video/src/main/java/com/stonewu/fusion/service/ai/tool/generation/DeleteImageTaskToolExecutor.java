package com.stonewu.fusion.service.ai.tool.generation;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/** 删除图片生成任务工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DeleteImageTaskToolExecutor implements ToolExecutor {

    private final ImageGenerationService imageGenerationService;

    @Override
    public String getToolName() {
        return "delete_image_task";
    }

    @Override
    public String getDisplayName() {
        return "删除图片生成任务";
    }

    @Override
    public String getToolDescription() {
        return "删除当前用户的图片生成任务及其生成结果，属于高风险操作。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "taskId": {"type": "integer", "description": "要删除的图片生成任务ID"}
                    },
                    "required": ["taskId"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long taskId = params.getLong("taskId");
            if (taskId == null) {
                return error("缺少 taskId");
            }
            imageGenerationService.delete(taskId, context.getUserId());
            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("deletedTaskId", taskId)
                    .toString();
        } catch (Exception e) {
            log.error("删除图片生成任务失败", e);
            return error("删除图片生成任务失败: " + e.getMessage());
        }
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
