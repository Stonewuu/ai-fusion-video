package com.stonewu.fusion.service.ai.tool.script;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.script.ScriptSceneItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 查询剧本单场次完整内容工具（get_script_scene）
 * <p>
 * 返回场次的完整信息：标头、环境描写、出场角色、完整对白/动作列表。
 * 用于分镜转换和子 Agent 场次解析时获取场次的全部细节。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class ScriptSceneItemDetailQueryToolExecutor implements ToolExecutor {

    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "get_script_scene";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询剧本场次详情";
    }

    @Override
    public String getToolDescription() {
        return """
                查询剧本中单个场次的完整内容，包含场景标头、环境描写、完整对白/动作列表、出场角色信息。
                用于分镜转换时获取场次的全部细节。""";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "scriptSceneItemId": {
                            "type": "integer",
                            "description": "剧本场次ID"
                        }
                    },
                    "required": ["scriptSceneItemId"]
                }""";
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long scriptSceneItemId = params.getLong("scriptSceneItemId");

            if (scriptSceneItemId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "参数错误：scriptSceneItemId 不能为空").toString();
            }

            ScriptSceneItem sceneItem = accessGuard.requireScriptScene(scriptSceneItemId, context.getUserId());

            // 构建返回结果
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("scriptSceneItemId", sceneItem.getId());
            result.put("scriptEpisodeId", sceneItem.getEpisodeId());
            result.put("sceneNumber", sceneItem.getSceneNumber());
            result.put("sceneHeading", sceneItem.getSceneHeading());
            result.put("location", sceneItem.getLocation());
            result.put("timeOfDay", sceneItem.getTimeOfDay());
            result.put("intExt", sceneItem.getIntExt());
            result.put("sceneDescription", sceneItem.getSceneDescription());

            // 出场角色
            if (sceneItem.getCharacters() != null) {
                result.put("characters", JSONUtil.parse(sceneItem.getCharacters()));
            }
            if (sceneItem.getCharacterAssetIds() != null) {
                result.put("characterAssetIds", JSONUtil.parse(sceneItem.getCharacterAssetIds()));
            }
            if (sceneItem.getSceneAssetId() != null) {
                result.put("sceneAssetId", sceneItem.getSceneAssetId());
            }
            if (sceneItem.getPropAssetIds() != null) {
                result.put("propAssetIds", JSONUtil.parse(sceneItem.getPropAssetIds()));
            }

            // 完整对白/动作列表
            if (sceneItem.getDialogues() != null) {
                result.put("dialogues", JSONUtil.parse(sceneItem.getDialogues()));
            }

            return JSONUtil.toJsonStr(result);

        } catch (Exception e) {
            log.error("查询场次详情失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
