package com.stonewu.fusion.service.ai.tool;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.script.ScriptService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 查询剧本工具（get_script）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ScriptQueryToolExecutor implements ToolExecutor {

    private final ScriptService scriptService;

    @Override
    public String getToolName() {
        return "get_script";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询剧本";
    }

    @Override
    public String getToolDescription() {
        return "查询剧本的基本信息（不含分集详情，如需分集请使用 get_script_structure）。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "scriptId": { "type": "number", "description": "剧本ID" }
                    },
                    "required": ["scriptId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long scriptId = params.getLong("scriptId");
            if (scriptId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 scriptId").toString();
            }

            Script script = scriptService.getById(scriptId);
            return JSONUtil.createObj()
                    .set("scriptId", script.getId())
                    .set("projectId", script.getProjectId())
                    .set("title", script.getTitle())
                    .set("content", script.getContent())
                    .set("storySynopsis", script.getStorySynopsis())
                    .set("charactersJson", script.getCharactersJson())
                    .set("totalEpisodes", script.getTotalEpisodes())
                    .set("parsingStatus", script.getParsingStatus())
                    .toString();
        } catch (Exception e) {
            log.error("查询剧本失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
