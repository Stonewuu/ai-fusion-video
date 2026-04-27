package com.stonewu.fusion.service.ai.provider;

import com.alibaba.cloud.ai.dashscope.api.DashScopeApi;
import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatModel;
import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatOptions;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.enums.ai.AiModelTypeEnum;
import com.stonewu.fusion.service.ai.dashscope.DashScopeGenerationSupport;
import com.stonewu.fusion.service.ai.proxy.AiProxySupport;
import io.agentscope.core.model.DashScopeChatModel.Builder;
import io.agentscope.core.model.Model;
import org.springframework.ai.chat.model.ChatModel;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * DashScope 提供商。
 */
@Component
@Slf4j
public class DashScopeAiProvider extends AbstractAiProvider {

    @Override
    public boolean supports(String platform) {
        return platform != null && "dashscope".equalsIgnoreCase(platform);
    }

    @Override
    public ChatModel createChatModel(AiProviderContext context) {
        String apiKey = context.getApiKey();
        Map<String, Object> config = context.getConfig();
        String modelName = context.getModelName();

        requireApiKey(apiKey, "DashScope");

        DashScopeChatOptions options = DashScopeChatOptions.builder()
                .model(modelName)
                .build();
        applyDouble(config, "temperature", options::setTemperature);
        applyDouble(config, "topP", options::setTopP);
        applyInt(config, "maxTokens", options::setMaxTokens);

        return DashScopeChatModel.builder()
                .dashScopeApi(DashScopeApi.builder()
                        .baseUrl(DashScopeGenerationSupport.resolveRootBaseUrl(context.getBaseUrl()))
                        .apiKey(apiKey)
                .restClientBuilder(AiProxySupport.restClientBuilder(
                    context.getApiConfig(), 60 * 1000, 3 * 60 * 1000))
                .webClientBuilder(AiProxySupport.webClientBuilder(
                    context.getApiConfig(), "dashscope-provider", Duration.ofSeconds(60)))
                        .build())
                .defaultOptions(options)
                .build();
    }

    @Override
    public Model createAgentScopeModel(AiProviderContext context) {
        Builder builder = io.agentscope.core.model.DashScopeChatModel.builder()
                .apiKey(context.getApiKey())
                .modelName(context.getModelName())
                .baseUrl(DashScopeGenerationSupport.resolveRootBaseUrl(context.getBaseUrl()))
                .stream(true);
        if (isReasoningEnabled(context)) {
            builder.enableThinking(true);
        }
        io.agentscope.core.model.transport.HttpTransport proxyTransport =
                AiProxySupport.agentScopeHttpTransport(context.getApiConfig());
        if (proxyTransport != null) {
            builder.httpTransport(proxyTransport);
        }
        return builder.build();
    }

    @Override
    public List<RemoteModelVO> listRemoteModels(AiProviderContext context) {
        requireApiKey(context.getApiKey(), "DashScope");
        String baseUrl = resolveApiBaseUrl(context.getBaseUrl());
        String url = joinUrl(baseUrl, "/models");
        log.info("[DashScopeAiProvider] 获取远程模型列表: {}", url);
        String response = executeGet(url, Map.of("Authorization", "Bearer " + context.getApiKey()), context.getApiConfig());
        List<RemoteModelVO> models = new ArrayList<>(parseDataArrayModels(response, "dashscope"));
        models.forEach(model -> {
            if (model.getModelType() == null) {
                model.setModelType(DashScopeGenerationSupport.inferRemoteModelType(model.getId()));
            }
        });
        mergeKnownGenerationModels(models);
        return models;
    }

    private String resolveApiBaseUrl(String baseUrl) {
        return DashScopeGenerationSupport.resolveCompatibleBaseUrl(baseUrl);
    }

    private void mergeKnownGenerationModels(List<RemoteModelVO> models) {
        Set<String> existingIds = new LinkedHashSet<>();
        for (RemoteModelVO model : models) {
            if (model.getId() != null) {
                existingIds.add(model.getId());
            }
        }
        addKnownModels(models, existingIds, AiModelTypeEnum.IMAGE.getType(),
                "wan2.7-image-pro",
                "wan2.7-image",
                "qwen-image-2.0-pro",
                "qwen-image-2.0",
                "qwen-image-max",
                "qwen-image-plus",
                "qwen-image");
        addKnownModels(models, existingIds, AiModelTypeEnum.VIDEO.getType(),
                "wan2.7-t2v",
                "wan2.7-i2v",
                "wan2.7-r2v",
                "wan2.7-videoedit",
                "wan2.6-t2v",
                "wan2.6-i2v",
                "wan2.6-i2v-flash",
                "wan2.2-kf2v-flash",
                "wanx2.1-t2v-plus",
                "wanx2.1-t2v-turbo",
                "wanx2.1-i2v-plus",
                "wanx2.1-i2v-turbo",
                "wanx2.1-kf2v-plus");
    }

    private void addKnownModels(List<RemoteModelVO> models, Set<String> existingIds, Integer modelType,
                                String... modelIds) {
        for (String modelId : modelIds) {
            if (existingIds.add(modelId)) {
                models.add(RemoteModelVO.builder()
                        .id(modelId)
                        .ownedBy("dashscope")
                        .modelType(modelType)
                        .build());
            }
        }
    }
}
