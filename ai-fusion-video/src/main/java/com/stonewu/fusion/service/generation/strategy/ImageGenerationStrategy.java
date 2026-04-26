package com.stonewu.fusion.service.generation.strategy;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;

import java.util.List;

/**
 * 图片生成策略接口
 * <p>
 * 不同平台（火山引擎/OpenAI 兼容/Vertex AI）实现此接口
 */
public interface ImageGenerationStrategy {

    /**
     * 策略名称（对应 ApiConfig.platform 值）
     */
    String getName();

    /**
     * 纯平台 API 调用：生成图片并返回 URL 列表
     * <p>
     * 此方法不涉及数据库操作，可被 AI 工具调用和异步队列共同复用。
     *
     * @param prompt    生图提示词
     * @param modelCode 模型代码（如 doubao-seedream-5-0-260128、dall-e-3）
     * @param width     图片宽度
     * @param height    图片高度
     * @param count     生成数量
     * @param imageUrls 参考图片 URL 列表（图生图时传入，文生图传 null 或空列表）
     * @param apiConfig API 配置（包含密钥、地址等）
     * @return 生成的图片 URL 列表
     */
    List<String> generate(String prompt, String modelCode, int width, int height, int count,
                          List<String> imageUrls, ApiConfig apiConfig);

    /**
     * 提交生图任务到平台（含数据库更新）
     *
     * @param task      生图任务
     * @param apiConfig API 配置（包含密钥、地址等）
     * @return 平台任务ID（同步平台可返回自定义标识）
     */
    String submit(ImageTask task, ApiConfig apiConfig);

    /**
     * 轮询平台任务状态（异步平台使用）
     * <p>
     * 同步 API 可在 submit 中直接完成，poll 置空实现即可
     *
     * @param platformTaskId 平台任务ID
     * @param task           生图任务
     * @param apiConfig      API 配置
     */
    void poll(String platformTaskId, ImageTask task, ApiConfig apiConfig);

    /**
     * 从模型配置中解析默认尺寸
     * <p>
     * 优先使用 task 中指定的尺寸，否则从 AiModel.config 的
     * defaultWidth/defaultHeight 读取，最终兜底 1024。
     *
     * @param model AiModel（可为 null）
     * @param task  生图任务
     * @return [width, height]
     */
    default int[] resolveDefaultSize(AiModel model, ImageTask task) {
        int width = (task.getWidth() != null && task.getWidth() > 0) ? task.getWidth() : 0;
        int height = (task.getHeight() != null && task.getHeight() > 0) ? task.getHeight() : 0;

        // 从模型 config 读取默认值
        if ((width <= 0 || height <= 0) && model != null && StrUtil.isNotBlank(model.getConfig())) {
            try {
                JSONObject config = JSONUtil.parseObj(model.getConfig());
                if (width <= 0) {
                    width = config.getInt("defaultWidth", 0);
                }
                if (height <= 0) {
                    height = config.getInt("defaultHeight", 0);
                }
            } catch (Exception ignored) {
                // config JSON 解析失败，使用兜底值
            }
        }

        // 最终兜底
        if (width <= 0) width = 1024;
        if (height <= 0) height = 1024;
        return new int[]{width, height};
    }

    /**
     * 从 ImageTask.refImageUrls JSON 字段解析参考图 URL 列表
     *
     * @param refImageUrls JSON 数组字符串，如 ["url1","url2"]，可为 null
     * @return URL 列表，无参考图时返回 null
     */
    default List<String> parseRefImageUrls(String refImageUrls) {
        if (StrUtil.isBlank(refImageUrls)) {
            return null;
        }
        try {
            List<String> urls = JSONUtil.parseArray(refImageUrls).toList(String.class);
            return urls.isEmpty() ? null : urls;
        } catch (Exception e) {
            return null;
        }
    }
}

