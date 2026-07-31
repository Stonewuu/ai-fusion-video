package com.stonewu.fusion.service.ai.tool.storyboard;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
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
import java.util.ArrayList;
import java.util.List;

/**
 * 保存场次分镜工具（save_storyboard_scene_shots）
 * <p>
 * 创建一个分镜场次，并批量插入该场次下的所有镜头。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SaveStoryboardSceneShotsToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "save_storyboard_scene_shots";
    }

    @Override
    public String getDisplayName() {
        return "保存场次分镜";
    }

    @Override
    public String getToolDescription() {
        return """
                创建一个分镜场次，并批量保存该场次下的所有镜头。
                可一次性保存一个场次的完整分镜数据。

                【参数说明】
                - storyboardId：所属分镜脚本ID（必填）
                - storyboardEpisodeId：所属分镜集ID（必填）
                - sceneNumber：场次编号，如 "1-1"（必填）
                - sceneHeading：场景标头，如 "内景 客厅 夜"
                - location：场景地点
                - timeOfDay：时间段（日/夜/黄昏/清晨）
                - intExt：内外景标识（内景/外景/内外景）
                - shots：镜头数组（必填），每个镜头含 content、shotType、duration、dialogue 等
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardId": {
                            "type": "integer",
                            "description": "所属分镜脚本ID（必填）"
                        },
                        "storyboardEpisodeId": {
                            "type": "integer",
                            "description": "所属分镜集ID（必填）"
                        },
                        "sceneNumber": {
                            "type": "string",
                            "description": "场次编号，如 1-1（必填）"
                        },
                        "sceneHeading": {
                            "type": "string",
                            "description": "场景标头，如 内景 客厅 夜"
                        },
                        "location": {
                            "type": "string",
                            "description": "场景地点"
                        },
                        "timeOfDay": {
                            "type": "string",
                            "description": "时间段（日/夜/黄昏/清晨）"
                        },
                        "intExt": {
                            "type": "string",
                            "description": "内外景标识（内景/外景/内外景）"
                        },
                        "sortOrder": {
                            "type": "integer",
                            "description": "场次排序序号（从0开始，默认0）"
                        },
                        "shots": {
                            "type": "array",
                            "description": "镜头列表（必填）",
                            "items": {
                                "type": "object",
                                "properties": {
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
                                        "description": "出场角色子资产ID列表（AssetItem.id），从 list_project_assets 或 query_asset_items 获得"
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
                                "required": ["content"]
                            }
                        }
                    },
                    "required": ["storyboardId", "storyboardEpisodeId", "sceneNumber", "shots"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long storyboardId = params.getLong("storyboardId");
            Long storyboardEpisodeId = params.getLong("storyboardEpisodeId");
            String sceneNumber = params.getStr("sceneNumber");
            JSONArray shotsArr = params.getJSONArray("shots");

            if (storyboardId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 storyboardId").toString();
            }
            if (storyboardEpisodeId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 storyboardEpisodeId").toString();
            }
            if (sceneNumber == null || sceneNumber.isBlank()) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 sceneNumber").toString();
            }
            if (shotsArr == null || shotsArr.isEmpty()) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 shots 或为空").toString();
            }
            accessGuard.requireStoryboardEpisode(storyboardId, storyboardEpisodeId, context.getUserId());

            StoryboardScene scene = StoryboardScene.builder()
                    .storyboardId(storyboardId)
                    .episodeId(storyboardEpisodeId)
                    .sceneNumber(sceneNumber)
                    .sceneHeading(params.getStr("sceneHeading"))
                    .location(params.getStr("location"))
                    .timeOfDay(params.getStr("timeOfDay"))
                    .intExt(params.getStr("intExt"))
                    .sortOrder(params.getInt("sortOrder", 0))
                    .status(1)
                    .build();

            List<StoryboardItem> items = new ArrayList<>();
            for (int i = 0; i < shotsArr.size(); i++) {
                JSONObject shot = shotsArr.getJSONObject(i);
                String content = shot.getStr("content");
                if (content == null || content.isBlank()) {
                    return JSONUtil.createObj().set("status", "error")
                            .set("message", "shots[" + i + "].content 不能为空").toString();
                }

                // 处理角色子资产ID列表
                String characterIdsJson = null;
                JSONArray charArray = shot.getJSONArray("characterIds");
                if (charArray != null) {
                    characterIdsJson = charArray.toString();
                }

                // 处理道具子资产ID列表
                String propIdsJson = null;
                JSONArray propArray = shot.getJSONArray("propIds");
                if (propArray != null) {
                    propIdsJson = propArray.toString();
                }

                StoryboardItem item = StoryboardItem.builder()
                        .storyboardId(storyboardId)
                        .storyboardEpisodeId(storyboardEpisodeId)
                        .sortOrder(i)
                        .shotNumber(String.valueOf(i + 1))
                        .shotType(shot.getStr("shotType"))
                        .content(content)
                        .sceneExpectation(shot.getStr("sceneExpectation"))
                        .dialogue(shot.getStr("dialogue"))
                        .sound(shot.getStr("sound"))
                        .soundEffect(shot.getStr("soundEffect"))
                        .music(shot.getStr("music"))
                        .cameraMovement(shot.getStr("cameraMovement"))
                        .cameraAngle(shot.getStr("cameraAngle"))
                        .cameraEquipment(shot.getStr("cameraEquipment"))
                        .focalLength(shot.getStr("focalLength"))
                        .transition(shot.getStr("transition"))
                        .characterIds(characterIdsJson)
                        .sceneAssetItemId(shot.getLong("sceneAssetItemId"))
                        .propIds(propIdsJson)
                        .remark(shot.getStr("remark"))
                        .duration(shot.get("duration") != null ? new BigDecimal(shot.getStr("duration")) : null)
                        .aiGenerated(true)
                        .status(1)
                        .build();
                items.add(item);
            }

            StoryboardScene savedScene = storyboardService.createSceneWithItems(scene, items);
            log.info("[save_storyboard_scene_shots] 场次及镜头保存成功: sceneId={}, count={}",
                    savedScene.getId(), items.size());

            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("storyboardSceneId", savedScene.getId())
                    .set("sceneNumber", sceneNumber)
                    .set("shotCount", items.size())
                    .set("message", "场次分镜保存成功，共 " + items.size() + " 个镜头").toString();
        } catch (Exception e) {
            log.error("保存场次分镜失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "操作失败: " + e.getMessage()).toString();
        }
    }
}
