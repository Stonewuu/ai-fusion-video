package com.stonewu.fusion.service.ai.tool;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * 查询分镜详情工具（get_storyboard）
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class StoryboardQueryToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;

    @Override
    public String getToolName() {
        return "get_storyboard";
    }

    @Override
    public boolean isReadOnly() {
        return true;
    }

    @Override
    public String getDisplayName() {
        return "查询分镜详情";
    }

    @Override
    public String getToolDescription() {
        return """
                查询分镜脚本的详情，包含所有分镜条目信息。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardId": {
                            "type": "number",
                            "description": "分镜ID"
                        }
                    },
                    "required": ["storyboardId"]
                }
                """;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long storyboardId = params.getLong("storyboardId");
            if (storyboardId == null) {
                return JSONUtil.createObj().set("status", "error").set("message", "缺少 storyboardId").toString();
            }

            Storyboard storyboard = storyboardService.getById(storyboardId);
            List<StoryboardEpisode> episodes = storyboardService.listEpisodes(storyboardId);
            List<StoryboardItem> items = storyboardService.listItems(storyboardId);

            JSONArray episodeList = new JSONArray();
            for (StoryboardEpisode episode : episodes) {
                long episodeItemCount = items.stream()
                        .filter(item -> episode.getId().equals(item.getStoryboardEpisodeId()))
                        .count();
                episodeList.add(JSONUtil.createObj()
                        .set("storyboardEpisodeId", episode.getId())
                        .set("scriptEpisodeId", episode.getScriptEpisodeId())
                        .set("episodeNumber", episode.getEpisodeNumber())
                        .set("title", episode.getTitle())
                        .set("synopsis", episode.getSynopsis())
                        .set("itemCount", episodeItemCount));
            }

            JSONArray itemList = new JSONArray();
            for (StoryboardItem item : items) {
                itemList.add(JSONUtil.createObj()
                        .set("id", item.getId())
                        .set("storyboardEpisodeId", item.getStoryboardEpisodeId())
                        .set("shotNumber", item.getShotNumber())
                        .set("autoShotNumber", item.getAutoShotNumber())
                        .set("shotType", item.getShotType())
                        .set("content", item.getContent())
                        .set("sceneExpectation", item.getSceneExpectation())
                        .set("dialogue", item.getDialogue())
                        .set("sound", item.getSound())
                        .set("duration", item.getDuration())
                        .set("cameraMovement", item.getCameraMovement())
                        .set("cameraAngle", item.getCameraAngle())
                        .set("transition", item.getTransition())
                        .set("imageUrl", item.getImageUrl())
                        .set("generatedImageUrl", item.getGeneratedImageUrl())
                        .set("firstFrameImageUrl", item.getFirstFrameImageUrl())
                        .set("lastFrameImageUrl", item.getLastFrameImageUrl())
                        .set("firstFramePrompt", item.getFirstFramePrompt())
                        .set("lastFramePrompt", item.getLastFramePrompt())
                        .set("videoUrl", item.getVideoUrl())
                        .set("generatedVideoUrl", item.getGeneratedVideoUrl())
                        .set("videoPrompt", item.getVideoPrompt()));
            }

            return JSONUtil.createObj()
                    .set("storyboardId", storyboard.getId())
                    .set("title", storyboard.getTitle())
                    .set("description", storyboard.getDescription())
                    .set("episodes", episodeList)
                    .set("totalItems", items.size())
                    .set("items", itemList)
                    .toString();
        } catch (Exception e) {
            log.error("查询分镜详情失败", e);
            return JSONUtil.createObj().set("status", "error").set("message", "查询失败: " + e.getMessage()).toString();
        }
    }
}
