package com.stonewu.fusion.service.generation.video.strategy.support;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;

/**
 * OpenAI 兼容视频协议适配上下文。
 */
public record OpenAiCompatibleVideoProtocolContext(
        AiModel model,
        ApiConfig apiConfig,
        VideoTask task,
        JSONObject modelConfig,
        AiModelMetadata metadata
) {
}
