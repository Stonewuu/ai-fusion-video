package com.stonewu.fusion.service.ai.tool.storyboard;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/** 通用更新分镜镜头工具。 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateStoryboardItemToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;
    private final ProjectService projectService;

    @Override
    public String getToolName() {
        return "update_storyboard_item";
    }

    @Override
    public String getDisplayName() {
        return "更新分镜镜头";
    }

    @Override
    public String getToolDescription() {
        return "通用更新分镜镜头的画面、声音、对白、运镜、素材关联、排序和状态字段。首尾帧可继续使用专用工具。";
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardItemId": {"type": "integer", "description": "分镜镜头ID"},
                        "sortOrder": {"type": "integer", "description": "排序值"},
                        "shotNumber": {"type": "string", "description": "镜头编号"},
                        "imageUrl": {"type": "string", "description": "镜头图片URL"},
                        "referenceImageUrl": {"type": "string", "description": "参考图片URL"},
                        "videoUrl": {"type": "string", "description": "镜头视频URL"},
                        "generatedImageUrl": {"type": "string", "description": "生成图片URL"},
                        "generatedVideoUrl": {"type": "string", "description": "生成视频URL"},
                        "videoPrompt": {"type": "string", "description": "视频提示词"},
                        "shotType": {"type": "string", "description": "景别"},
                        "duration": {"type": "number", "description": "镜头时长"},
                        "content": {"type": "string", "description": "画面内容"},
                        "sceneExpectation": {"type": "string", "description": "场景预期"},
                        "sound": {"type": "string", "description": "声音描述"},
                        "dialogue": {"type": "string", "description": "对白"},
                        "soundEffect": {"type": "string", "description": "音效"},
                        "music": {"type": "string", "description": "音乐"},
                        "cameraMovement": {"type": "string", "description": "运镜"},
                        "cameraAngle": {"type": "string", "description": "机位角度"},
                        "cameraEquipment": {"type": "string", "description": "摄影设备"},
                        "focalLength": {"type": "string", "description": "焦距"},
                        "transition": {"type": "string", "description": "转场"},
                        "characterIds": {"type": "string", "description": "角色资产ID JSON"},
                        "sceneAssetItemId": {"type": "integer", "description": "场景子资产ID"},
                        "propIds": {"type": "string", "description": "道具资产ID JSON"},
                        "remark": {"type": "string", "description": "备注"},
                        "customData": {"type": "string", "description": "自定义数据JSON"},
                        "aiGenerated": {"type": "boolean", "description": "是否由AI生成"},
                        "status": {"type": "integer", "description": "状态"}
                    },
                    "required": ["storyboardItemId"],
                    "additionalProperties": false
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long itemId = params.getLong("storyboardItemId");
            if (itemId == null) return error("缺少 storyboardItemId");
            if (params.size() == 1) return error("至少提供一个需要更新的字段");

            StoryboardItem existing = storyboardService.getItemById(itemId);
            Storyboard storyboard = storyboardService.getById(existing.getStoryboardId());
            if (!projectService.canAccessProject(storyboard.getProjectId(), context.getUserId())) {
                return error("无权更新该分镜镜头");
            }

            StoryboardItem patch = new StoryboardItem();
            patch.setId(itemId);
            patch.setSortOrder(null);
            patch.setAiGenerated(null);
            patch.setStatus(null);
            applyFields(patch, params);
            StoryboardItem saved = storyboardService.updateItem(patch);
            return JSONUtil.createObj().set("status", "success").set("storyboardItem", saved).toString();
        } catch (Exception e) {
            log.error("更新分镜镜头失败", e);
            return error("更新分镜镜头失败: " + e.getMessage());
        }
    }

    private void applyFields(StoryboardItem item, JSONObject params) {
        if (params.containsKey("sortOrder")) item.setSortOrder(params.getInt("sortOrder"));
        if (params.containsKey("shotNumber")) item.setShotNumber(params.getStr("shotNumber"));
        if (params.containsKey("imageUrl")) item.setImageUrl(params.getStr("imageUrl"));
        if (params.containsKey("referenceImageUrl")) item.setReferenceImageUrl(params.getStr("referenceImageUrl"));
        if (params.containsKey("videoUrl")) item.setVideoUrl(params.getStr("videoUrl"));
        if (params.containsKey("generatedImageUrl")) {
            item.setGeneratedImageUrl(params.getStr("generatedImageUrl"));
        }
        if (params.containsKey("generatedVideoUrl")) {
            item.setGeneratedVideoUrl(params.getStr("generatedVideoUrl"));
        }
        if (params.containsKey("videoPrompt")) item.setVideoPrompt(params.getStr("videoPrompt"));
        if (params.containsKey("shotType")) item.setShotType(params.getStr("shotType"));
        if (params.containsKey("duration")) item.setDuration(params.getBigDecimal("duration"));
        if (params.containsKey("content")) item.setContent(params.getStr("content"));
        if (params.containsKey("sceneExpectation")) {
            item.setSceneExpectation(params.getStr("sceneExpectation"));
        }
        if (params.containsKey("sound")) item.setSound(params.getStr("sound"));
        if (params.containsKey("dialogue")) item.setDialogue(params.getStr("dialogue"));
        if (params.containsKey("soundEffect")) item.setSoundEffect(params.getStr("soundEffect"));
        if (params.containsKey("music")) item.setMusic(params.getStr("music"));
        if (params.containsKey("cameraMovement")) item.setCameraMovement(params.getStr("cameraMovement"));
        if (params.containsKey("cameraAngle")) item.setCameraAngle(params.getStr("cameraAngle"));
        if (params.containsKey("cameraEquipment")) {
            item.setCameraEquipment(params.getStr("cameraEquipment"));
        }
        if (params.containsKey("focalLength")) item.setFocalLength(params.getStr("focalLength"));
        if (params.containsKey("transition")) item.setTransition(params.getStr("transition"));
        if (params.containsKey("characterIds")) item.setCharacterIds(params.getStr("characterIds"));
        if (params.containsKey("sceneAssetItemId")) {
            item.setSceneAssetItemId(params.getLong("sceneAssetItemId"));
        }
        if (params.containsKey("propIds")) item.setPropIds(params.getStr("propIds"));
        if (params.containsKey("remark")) item.setRemark(params.getStr("remark"));
        if (params.containsKey("customData")) item.setCustomData(params.getStr("customData"));
        if (params.containsKey("aiGenerated")) item.setAiGenerated(params.getBool("aiGenerated"));
        if (params.containsKey("status")) item.setStatus(params.getInt("status"));
    }

    private String error(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
