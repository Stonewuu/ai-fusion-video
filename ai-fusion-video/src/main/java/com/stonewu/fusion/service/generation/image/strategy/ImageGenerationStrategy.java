package com.stonewu.fusion.service.generation.image.strategy;

import cn.hutool.core.util.StrUtil;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.generation.job.PreparedGenerationCall;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;

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
     * 一个意图对应一个远端请求；未声明原生批量的协议每次只请求一个输出。
     */
    default int batchSize(ImageTask task, AiModel model, ApiConfig apiConfig) {
        return 1;
    }

    /**
     * 准备所有素材与请求体，不发送生成请求；返回只能提交一次的调用。
     */
    PreparedGenerationCall prepare(ImageTask task, AiModel model, ApiConfig apiConfig);

    /**
     * 每次调度只查询一次，等待由持久化调度器负责。
     */
    RemoteGenerationResult query(String platformTaskId, ImageTask task, AiModel model, ApiConfig apiConfig);

    /**
     * true 仅表示远端接收取消请求，终态仍需查询确认。
     */
    default boolean cancel(String platformTaskId, ImageTask task, AiModel model, ApiConfig apiConfig) {
        return false;
    }

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
        if (width <= 0) {
            width = 1024;
        }
        if (height <= 0) {
            height = 1024;
        }
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

