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
 * 更新剧本信息工具（update_script_info）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateScriptInfoToolExecutor implements ToolExecutor {

    private final ScriptService scriptService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "update_script_info";
    }

    @Override
    public String getDisplayName() {
        return "更新剧本信息";
    }

    @Override
    public String getToolDescription() {
        return """
                更新剧本的总体信息，包括故事梗概(storySynopsis)和人物表(charactersJson)等字段。
                scriptId 必填，各字段均为可选，仅传入需要更新的字段。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "scriptId": { "type": "integer", "description": "剧本ID（必填）" },
                        "storySynopsis": { "type": "string", "description": "全剧故事梗概（200-500字概述）" },
                        "charactersJson": { "type": "array", "description": "人物表JSON快照", "items": { "type": "object", "properties": { "name": { "type": "string" }, "assetId": { "type": "integer" }, "description": { "type": "string" }, "importance": { "type": "string" } }, "required": ["name"] } },
                        "genre": { "type": "string", "description": "类型/风格" }
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

            Script script = accessGuard.requireScript(scriptId, context.getUserId());
            StringBuilder updatedFields = new StringBuilder();

            if (params.containsKey("storySynopsis")) {
                script.setStorySynopsis(params.getStr("storySynopsis"));
                updatedFields.append("故事梗概、");
            }
            if (params.containsKey("charactersJson")) {
                Object chars = params.get("charactersJson");
                script.setCharactersJson(chars instanceof String ? (String) chars : JSONUtil.toJsonStr(chars));
                updatedFields.append("人物表、");
            }
            if (params.containsKey("genre")) {
                script.setGenre(params.getStr("genre"));
                updatedFields.append("类型、");
            }

            if (updatedFields.isEmpty()) {
                return JSONUtil.createObj().set("status", "error").set("message", "未指定任何要更新的字段").toString();
            }

            scriptService.update(script);
            updatedFields.setLength(updatedFields.length() - 1);
            return JSONUtil.createObj()
                    .set("scriptId", scriptId)
                    .set("message", "已更新剧本信息：" + updatedFields).toString();
        } catch (Exception e) {
            log.error("更新剧本信息失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "更新失败: " + e.getMessage()).toString();
        }
    }
}
