package com.stonewu.fusion.service.ai.tool.storyboard;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 更新分镜条目视频URL工具（update_storyboard_item_video）
 * <p>
 * 将生成的视频 URL 回填到分镜条目的 generatedVideoUrl 字段。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateStoryboardItemVideoToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;
    private final ToolResourceAccessGuard accessGuard;

    @Override
    public String getToolName() {
        return "update_storyboard_item_video";
    }

    @Override
    public String getDisplayName() {
        return "更新分镜视频";
    }

    @Override
    public String getToolDescription() {
        return """
                将生成的视频URL保存到分镜条目中。
                调用 generate_video 获取视频URL后，使用此工具将视频关联到对应的分镜镜头。
                """;
    }

    @Override
    public String getParametersSchema() {
        return """
                {
                    "type": "object",
                    "properties": {
                        "storyboardItemId": {
                            "type": "integer",
                            "description": "分镜条目ID"
                        },
                        "videoUrl": {
                            "type": "string",
                            "description": "生成的视频URL"
                        },
                        "videoPrompt": {
                            "type": "string",
                            "description": "生成视频时使用的提示词（保存以便后续复用或手动调整）"
                        },
                        "coverUrl": {
                            "type": "string",
                            "description": "视频封面图URL（可选）"
                        }
                    },
                    "required": ["storyboardItemId"]
                }
                """;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }

    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long itemId = params.getLong("storyboardItemId");
            String videoUrl = params.getStr("videoUrl");

            if (itemId == null) {
                return errorResult("缺少 storyboardItemId");
            }

            StoryboardItem item = accessGuard.requireStoryboardItem(itemId, context.getUserId());

            // 保存视频URL（promptOnly 模式下可能为空）
            if (StrUtil.isNotBlank(videoUrl)) {
                item.setGeneratedVideoUrl(videoUrl);
            }

            // 保存生成时使用的提示词
            String videoPrompt = params.getStr("videoPrompt");
            if (StrUtil.isNotBlank(videoPrompt)) {
                item.setVideoPrompt(videoPrompt);
            }

            // 至少需要传入 videoUrl 或 videoPrompt 之一
            if (StrUtil.isBlank(videoUrl) && StrUtil.isBlank(videoPrompt)) {
                return errorResult("videoUrl 和 videoPrompt 至少需要传入一个");
            }

            storyboardService.updateItem(item);

            log.info("[update_storyboard_item_video] 已更新分镜视频: itemId={}, videoUrl={}",
                    itemId, videoUrl);

            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("storyboardItemId", itemId)
                    .set("videoUrl", videoUrl)
                    .set("message", "视频已保存到分镜条目")
                    .toString();

        } catch (Exception e) {
            log.error("[update_storyboard_item_video] 更新失败", e);
            return errorResult("更新失败: " + e.getMessage());
        }
    }

    private String errorResult(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
