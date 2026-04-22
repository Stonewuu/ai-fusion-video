package com.stonewu.fusion.service.ai.provider;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.service.ai.googleflow.GoogleFlowReverseApiSupport;
import io.agentscope.core.model.Model;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Google Flow Reverse API 提供商。
 * <p>
 * 主要用于远程拉取可用的别名模型列表，避免把具体分辨率型号直接暴露到内部配置层。
 */
@Component
public class GoogleFlowReverseApiProvider extends AbstractAiProvider {

    @Override
    public boolean supports(String platform) {
        return GoogleFlowReverseApiSupport.PLATFORM.equalsIgnoreCase(platform);
    }

    @Override
    public ChatModel createChatModel(AiProviderContext context) {
        throw new BusinessException("GoogleFlowReverseApi 仅支持图片/视频生成模型，不支持对话模型");
    }

    @Override
    public Model createAgentScopeModel(AiProviderContext context) {
        throw new BusinessException("GoogleFlowReverseApi 仅支持图片/视频生成模型，不支持 AgentScope 对话模型");
    }

    @Override
    public List<RemoteModelVO> listRemoteModels(AiProviderContext context) {
        requireApiKey(context.getApiKey(), GoogleFlowReverseApiSupport.PLATFORM);
        String baseUrl = GoogleFlowReverseApiSupport.resolveBaseUrl(context.getBaseUrl());

        String json = executeGet(joinUrl(baseUrl, "/v1/models/aliases"),
                java.util.Map.of("Authorization", "Bearer " + context.getApiKey()));

        return parseDataArrayModels(json, "flow2api").stream()
                .filter(model -> !model.getId().startsWith("veo_3_1_upsampler_"))
            .map(model -> RemoteModelVO.builder()
                .id(model.getId())
                .ownedBy(model.getOwnedBy())
                .modelType(GoogleFlowReverseApiSupport.inferRemoteModelType(model.getId()))
                .build())
                .toList();
    }
}