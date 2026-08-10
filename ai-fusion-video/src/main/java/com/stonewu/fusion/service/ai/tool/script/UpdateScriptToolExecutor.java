package com.stonewu.fusion.service.ai.tool.script;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import com.stonewu.fusion.service.script.ScriptService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 更新剧本工具（update_script）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateScriptToolExecutor implements ToolExecutor {

    private final ScriptService scriptService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "update_script";
    }

    @Override
    public String getDisplayName() {
        return "更新剧本正文";
    }

    @Override
    public String getToolDescription() {
        return "仅更新剧本正文内容。故事梗概、人物表和类型请使用 update_script_info；顶层剧本名称固定跟随项目名称。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "scriptId": { "type": "integer", "description": "剧本ID" },
                        "content": { "type": "string", "description": "剧本正文内容" }
                    },
                    "required": ["scriptId", "content"]
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

            if (!params.containsKey("content")) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 content").toString();
            }

            Script script = accessGuard.requireScript(scriptId, context.getUserId());
            script.setContent(params.getStr("content"));
            scriptService.update(script);
            return JSONUtil.createObj()
                    .set("scriptId", scriptId)
                    .set("message", "剧本正文已更新").toString();
        } catch (Exception e) {
            log.error("更新剧本失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "更新失败: " + e.getMessage()).toString();
        }
    }
}
