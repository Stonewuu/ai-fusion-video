package com.stonewu.fusion.service.generation.video.strategy.newapi;

import cn.hutool.json.JSONObject;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.model.AiModelMetadata;

/**
 * NewAPI 视频协议适配上下文。
 */
public record NewApiVideoProtocolContext(
        AiModel model,
        ApiConfig apiConfig,
        VideoTask task,
        JSONObject modelConfig,
        AiModelMetadata metadata
) {
}
