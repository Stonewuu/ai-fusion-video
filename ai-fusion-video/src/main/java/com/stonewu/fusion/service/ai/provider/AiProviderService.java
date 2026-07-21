package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.ai.model.RemoteModelMetadata;
import io.agentscope.core.model.ChatModelBase;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 提供统一的提供商调用入口。
 */
@Service
@RequiredArgsConstructor
public class AiProviderService {

    private final AiProviderContextFactory contextFactory;
    private final AiProviderRegistry providerRegistry;
    private final AiModelMetadataResolver aiModelMetadataResolver;

    public ChatModel createChatModel(AiModel model) {
        AiProviderContext context = contextFactory.createForModel(model);
        return providerRegistry.getProvider(context).createChatModel(context);
    }

    public ChatModelBase createAgentScopeModel(AiModel model) {
        AiProviderContext context = contextFactory.createForModel(model);
        return providerRegistry.getProvider(context).createAgentScopeModel(context);
    }

    public List<RemoteModelVO> listRemoteModels(Long apiConfigId) {
        AiProviderContext context = contextFactory.createForApiConfig(apiConfigId);
        return providerRegistry.getProvider(context).listRemoteModels(context).stream()
            .map(model -> enrichRemoteModel(context.getPlatform(), model))
            .toList();
        }

        private RemoteModelVO enrichRemoteModel(String providerPlatform, RemoteModelVO model) {
        RemoteModelMetadata metadata = aiModelMetadataResolver.resolveRemoteModel(
            providerPlatform,
            model.getId(),
            StrUtil.blankToDefault(model.getDisplayName(), model.getOwnedBy()),
            model.getModelType());

        return RemoteModelVO.builder()
            .id(model.getId())
            .displayName(StrUtil.blankToDefault(model.getDisplayName(), metadata.displayName()))
            .ownedBy(model.getOwnedBy())
            .providerPlatform(StrUtil.blankToDefault(model.getProviderPlatform(), metadata.providerPlatform()))
            .modelType(model.getModelType() != null ? model.getModelType() : metadata.modelType())
            .modelFamily(StrUtil.blankToDefault(model.getModelFamily(), metadata.modelFamily()))
            .modelProtocol(StrUtil.blankToDefault(model.getModelProtocol(), metadata.modelProtocol()))
            .inferredMetadata(model.getInferredMetadata() != null ? model.getInferredMetadata() : metadata.inferred())
            .build();
    }
}
