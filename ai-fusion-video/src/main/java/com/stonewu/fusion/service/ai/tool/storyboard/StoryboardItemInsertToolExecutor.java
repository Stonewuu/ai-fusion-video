package com.stonewu.fusion.service.ai.tool.storyboard;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.entity.storyboard.StoryboardScene;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.util.List;

/**
 * 插入分镜条目工具（insert_storyboard_item）
 * <p>
 * 在指定分镜中插入新的分镜镜头。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class StoryboardItemInsertToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "insert_storyboard_item";
    }

    @Override
    public String getDisplayName() {
        return "插入分镜条目";
    }

    @Override
    public String getToolDescription() {
        return """
                在指定分镜场次中插入新的镜头，自动继承所属分镜和分镜集并添加到场次末尾。

                【使用提示】
                - storyboardSceneId 为必填参数，必须使用已有分镜场次ID
                - content 为必填参数，描述画面内容
                - 可指定景别、时长、对白、音效、配乐、镜头运动等详细信息
                - 可关联角色资产ID和场景资产ID
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardSceneId": {
                            "type": "integer",
                            "description": "镜头所属的分镜场次ID（必填）"
                        },
                        "content": {
                            "type": "string",
                            "description": "画面内容描述（必填）"
                        },
                        "shotType": {
                            "type": "string",
                            "description": "景别（远景/全景/中景/近景/特写）"
                        },
                        "duration": {
                            "type": "number",
                            "description": "预估时长（秒）"
                        },
                        "dialogue": {
                            "type": "string",
                            "description": "台词/旁白，格式为'角色名：台词内容'，如'张三：你好啊'。旁白则直接写内容，无需角色名前缀"
                        },
                        "sound": {
                            "type": "string",
                            "description": "声音描述"
                        },
                        "soundEffect": {
                            "type": "string",
                            "description": "音效"
                        },
                        "music": {
                            "type": "string",
                            "description": "配乐建议"
                        },
                        "cameraMovement": {
                            "type": "string",
                            "description": "镜头运动（推/拉/摇/移等）"
                        },
                        "cameraAngle": {
                            "type": "string",
                            "description": "镜头角度（平视/俯视/仰视等）"
                        },
                        "cameraEquipment": {
                            "type": "string",
                            "description": "摄像机装备"
                        },
                        "focalLength": {
                            "type": "string",
                            "description": "镜头焦段"
                        },
                        "transition": {
                            "type": "string",
                            "description": "转场效果"
                        },
                        "sceneExpectation": {
                            "type": "string",
                            "description": "画面期望描述"
                        },
                        "characterIds": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "出场角色子资产ID列表（AssetItem.id）"
                        },
                        "sceneAssetItemId": {
                            "type": "integer",
                            "description": "场景子资产ID（AssetItem.id）"
                        },
                        "propIds": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "道具子资产ID列表（AssetItem.id）"
                        },
                        "remark": {
                            "type": "string",
                            "description": "备注"
                        }
                    },
                    "required": ["storyboardSceneId", "content"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long storyboardSceneId = params.getLong("storyboardSceneId");
            if (storyboardSceneId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 storyboardSceneId").toString();
            }
            String content = params.getStr("content");
            if (content == null || content.isBlank()) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 content").toString();
            }

            StoryboardScene scene = accessGuard.requireStoryboardScene(storyboardSceneId, context.getUserId());
            StoryboardEpisode episode = accessGuard.requireStoryboardEpisode(
                    scene.getStoryboardId(), scene.getEpisodeId(), context.getUserId());
            List<StoryboardItem> existing = storyboardService.listItemsByScene(storyboardSceneId);
            int nextOrder = existing.isEmpty() ? 0 : existing.get(existing.size() - 1).getSortOrder() + 1;

            // 处理角色子资产ID列表
            String characterIdsJson = null;
            JSONArray charArray = params.getJSONArray("characterIds");
            if (charArray != null) {
                characterIdsJson = charArray.toString();
            }

            // 处理道具子资产ID列表
            String propIdsJson = null;
            JSONArray propArray = params.getJSONArray("propIds");
            if (propArray != null) {
                propIdsJson = propArray.toString();
            }

            StoryboardItem item = StoryboardItem.builder()
                    .storyboardId(scene.getStoryboardId())
                    .storyboardEpisodeId(episode.getId())
                    .storyboardSceneId(storyboardSceneId)
                    .sortOrder(nextOrder)
                    .shotNumber(params.getStr("shotNumber"))
                    .shotType(params.getStr("shotType"))
                    .content(content)
                    .sceneExpectation(params.getStr("sceneExpectation"))
                    .dialogue(params.getStr("dialogue"))
                    .sound(params.getStr("sound"))
                    .soundEffect(params.getStr("soundEffect"))
                    .music(params.getStr("music"))
                    .cameraMovement(params.getStr("cameraMovement"))
                    .cameraAngle(params.getStr("cameraAngle"))
                    .cameraEquipment(params.getStr("cameraEquipment"))
                    .focalLength(params.getStr("focalLength"))
                    .transition(params.getStr("transition"))
                    .characterIds(characterIdsJson)
                    .sceneAssetItemId(params.getLong("sceneAssetItemId"))
                    .propIds(propIdsJson)
                    .remark(params.getStr("remark"))
                    .duration(params.get("duration") != null ? new BigDecimal(params.getStr("duration")) : null)
                    .build();

            StoryboardItem saved = storyboardService.createItem(item);
            return JSONUtil.createObj()
                    .set("itemId", saved.getId())
                    .set("sortOrder", saved.getSortOrder())
                    .set("message", "分镜条目添加成功").toString();
        } catch (Exception e) {
            log.error("插入分镜条目失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "操作失败: " + e.getMessage()).toString();
        }
    }
}
