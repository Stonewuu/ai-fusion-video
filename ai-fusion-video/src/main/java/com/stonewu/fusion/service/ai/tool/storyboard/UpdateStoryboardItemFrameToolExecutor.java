package com.stonewu.fusion.service.ai.tool.storyboard;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONUtil;
import cn.hutool.json.JSONObject;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.tool.ToolResourceAccessGuard;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

/**
 * 更新分镜条目首尾帧工具（update_storyboard_item_frame）。
 * <p>
 * 将 AI 生成的图片 URL 回填到分镜条目的首帧或尾帧字段。
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class UpdateStoryboardItemFrameToolExecutor implements ToolExecutor {

    private final StoryboardService storyboardService;
    private final ToolResourceAccessGuard accessGuard;

    /**
     * 获取工具名称。
     *
     * @return 工具名称
     */
    @Override
    public String getToolName() {
        return "update_storyboard_item_frame";
    }

    /**
     * 获取工具展示名称。
     *
     * @return 工具展示名称
     */
    @Override
    public String getDisplayName() {
        return "更新分镜首尾帧";
    }

    /**
     * 获取工具说明。
     *
     * @return 工具说明
     */
    @Override
    public String getToolDescription() {
        return """
                将生成的图片URL保存到分镜条目的首帧或尾帧字段中。
                调用 generate_image 获取图片URL后，使用此工具将图片关联到对应的分镜镜头。
                """;
    }

    /**
     * 获取工具参数 JSON Schema。
     *
     * @return 参数 JSON Schema
     */
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
                        "frameType": {
                            "type": "string",
                            "enum": ["first", "last"],
                            "description": "帧类型：first-首帧，last-尾帧"
                        },
                        "imageUrl": {
                            "type": "string",
                            "description": "generate_image 返回的图片URL"
                        },
                        "framePrompt": {
                            "type": "string",
                            "description": "生成首尾帧时使用的提示词"
                        }
                    },
                    "required": ["storyboardItemId", "frameType", "imageUrl"]
                }
                """;
    }

    /**
     * 判断工具是否启用。
     *
     * @return 是否启用
     */
    @Override
    public boolean isEnabled() {
        return true;
    }

    /**
     * 执行分镜首尾帧回填。
     *
     * @param toolInput 工具入参 JSON
     * @param context   工具执行上下文
     * @return 执行结果 JSON
     */
    @Override
    public String execute(String toolInput, ToolExecutionContext context) {
        try {
            JSONObject params = JSONUtil.parseObj(toolInput);
            Long itemId = params.getLong("storyboardItemId");
            String frameType = params.getStr("frameType");
            String imageUrl = params.getStr("imageUrl");
            String framePrompt = params.getStr("framePrompt");

            if (itemId == null) {
                return errorResult("缺少 storyboardItemId");
            }
            if (StrUtil.isBlank(frameType)) {
                return errorResult("缺少 frameType");
            }
            if (StrUtil.isBlank(imageUrl)) {
                return errorResult("缺少 imageUrl");
            }

            accessGuard.requireStoryboardItem(itemId, context.getUserId());
            StoryboardItem updated = storyboardService.updateItemFrame(itemId, frameType, imageUrl, framePrompt);

            log.info("[update_storyboard_item_frame] 已更新分镜首尾帧: itemId={}, frameType={}, imageUrl={}",
                    itemId, frameType, imageUrl);

            return JSONUtil.createObj()
                    .set("status", "success")
                    .set("storyboardItemId", itemId)
                    .set("frameType", storyboardService.normalizeFrameType(frameType))
                    .set("imageUrl", imageUrl)
                    .set("firstFrameImageUrl", updated.getFirstFrameImageUrl())
                    .set("lastFrameImageUrl", updated.getLastFrameImageUrl())
                    .set("message", "分镜首尾帧已保存")
                    .toString();
        } catch (Exception e) {
            log.error("[update_storyboard_item_frame] 更新分镜首尾帧失败", e);
            return errorResult("更新失败: " + e.getMessage());
        }
    }

    /**
     * 构建错误结果。
     *
     * @param message 错误消息
     * @return 错误结果 JSON
     */
    private String errorResult(String message) {
        return JSONUtil.createObj().set("status", "error").set("message", message).toString();
    }
}
