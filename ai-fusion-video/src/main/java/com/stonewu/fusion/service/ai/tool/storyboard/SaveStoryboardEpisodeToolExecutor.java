package com.stonewu.fusion.service.ai.tool.storyboard;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 保存分镜集工具（save_storyboard_episode）
 * <p>
 * 在指定分镜脚本下按剧本分集创建或复用分镜集记录。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SaveStoryboardEpisodeToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "save_storyboard_episode";
    }

    @Override
    public String getDisplayName() {
        return "保存分镜集";
    }

    @Override
    public String getToolDescription() {
        return """
                在指定分镜脚本下按剧本分集创建或复用分镜集记录。
                返回分镜集ID，后续可为该分镜集添加场次和镜头。

                【参数说明】
                - storyboardId：所属分镜脚本ID（必填）
                - scriptEpisodeId：关联的剧本分集ID（必填）
                - episodeNumber：集号，从1开始（必填）
                - title：本集标题（必填）
                - synopsis：本集梗概（可选）
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
                        "scriptEpisodeId": {
                            "type": "integer",
                            "description": "关联的剧本分集ID（必填）"
                        },
                        "episodeNumber": {
                            "type": "integer",
                            "description": "集号，从1开始（必填）"
                        },
                        "title": {
                            "type": "string",
                            "description": "本集标题（必填）"
                        },
                        "synopsis": {
                            "type": "string",
                            "description": "本集梗概"
                        }
                    },
                    "required": ["storyboardId", "scriptEpisodeId", "episodeNumber", "title"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long storyboardId = params.getLong("storyboardId");
            Long scriptEpisodeId = params.getLong("scriptEpisodeId");
            Integer episodeNumber = params.getInt("episodeNumber");
            String title = params.getStr("title");

            if (storyboardId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 storyboardId").toString();
            }
            if (scriptEpisodeId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 scriptEpisodeId").toString();
            }
            if (episodeNumber == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 episodeNumber").toString();
            }
            if (title == null || title.isBlank()) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 title").toString();
            }

            accessGuard.requireStoryboard(storyboardId, context.getUserId());
            StoryboardEpisode saved = storyboardService.saveEpisodeForScript(
                    storyboardId,
                    scriptEpisodeId,
                    episodeNumber,
                    title,
                    params.getStr("synopsis"));
            log.info("[save_storyboard_episode] 分镜集保存成功: id={}, storyboardId={}, scriptEpisodeId={}, title={}",
                    saved.getId(), storyboardId, scriptEpisodeId, title);

            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("storyboardEpisodeId", saved.getId())
                    .set("storyboardId", storyboardId)
                    .set("scriptEpisodeId", scriptEpisodeId)
                    .set("episodeNumber", episodeNumber)
                    .set("title", title)
                    .set("message", "分镜集保存成功").toString();
        } catch (Exception e) {
            log.error("保存分镜集失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "操作失败: " + e.getMessage()).toString();
        }
    }
}
