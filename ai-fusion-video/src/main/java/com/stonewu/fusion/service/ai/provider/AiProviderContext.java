package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import lombok.Builder;
import lombok.Value;

import java.util.Map;

/**
 * 提供商执行时需要的统一上下文。
 */
@Value
@Builder(toBuilder = true)
public class AiProviderContext {

    AiModel model;

    ApiConfig apiConfig;

    @Builder.Default
    Map<String, Object> config = Map.of();

    String platform;

    String apiKey;

    String baseUrl;

    String modelName;

    /**
     * 持久化任务的一次提交；HTTP 与 SDK 均不得自行重复发送。
     */
    boolean singleRequest;
}
