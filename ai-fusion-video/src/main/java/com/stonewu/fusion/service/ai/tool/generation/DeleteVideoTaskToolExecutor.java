package com.stonewu.fusion.service.ai.tool.generation;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/** 删除视频生成任务工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class DeleteVideoTaskToolExecutor implements ToolExecutor {

    private final VideoGenerationService videoGenerationService;

    @Override
    public String getToolName() {
        return "delete_video_task";
    }

    @Override
    public String getDisplayName() {
        return "删除视频生成任务";
    }

    @Override
    public String getToolDescription() {
        return "删除当前用户的视频生成任务及其生成结果，属于高风险操作。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "taskId": {"type": "integer", "description": "要删除的视频生成任务ID"}
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
            videoGenerationService.delete(taskId, context.getUserId());
            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("deletedTaskId", taskId)
                    .toString();
        } catch (Exception e) {
            log.error("删除视频生成任务失败", e);
            return error("删除视频生成任务失败: " + e.getMessage());
        }
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
