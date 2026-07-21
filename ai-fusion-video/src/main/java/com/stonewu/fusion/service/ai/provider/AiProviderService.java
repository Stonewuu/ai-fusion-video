package com.stonewu.fusion.service.ai.provider;

import cn.hutool.core.util.StrUtil;
import com.stonewu.fusion.controller.ai.vo.RemoteModelVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelKey;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.ai.model.RemoteModelMetadata;
import io.agentscope.core.model.ChatModelBase;
import lombok.RequiredArgsConstructor;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Objects;

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
        return provisionAgentScopeModel(model).model();
    }

    public String agentScopeModelFingerprint(AiModel model) {
        return fingerprint(contextFactory.createForModel(model));
    }

    public AgentScopeModelProvision provisionAgentScopeModel(AiModel model) {
        AiProviderContext context = contextFactory.createForModel(model);
        ChatModelBase agentScopeModel = providerRegistry.getProvider(context)
                .createAgentScopeModel(context);
        return new AgentScopeModelProvision(agentScopeModel, fingerprint(context));
    }

    private String fingerprint(AiProviderContext context) {
        AiModel model = Objects.requireNonNull(context.getModel(), "provider model must not be null");
        ApiConfig apiConfig = context.getApiConfig();
        return AgentKernelKey.contentFingerprint(
                "agentscope-provider-config-v1",
                AgentKernelKey.modelRecordFingerprint(model),
                context.getPlatform(),
                context.getApiKey(),
                context.getBaseUrl(),
                context.getModelName(),
                apiConfig != null ? apiConfig.getId() : null,
                apiConfig != null ? apiConfig.getPlatform() : null,
                apiConfig != null ? apiConfig.getApiType() : null,
                apiConfig != null ? apiConfig.getApiUrl() : null,
                apiConfig != null ? apiConfig.getAutoAppendV1Path() : null,
                apiConfig != null ? apiConfig.getProxyType() : null,
                apiConfig != null ? apiConfig.getProxyHost() : null,
                apiConfig != null ? apiConfig.getProxyPort() : null,
                apiConfig != null ? apiConfig.getProxyUsername() : null,
                apiConfig != null ? apiConfig.getProxyPassword() : null,
                apiConfig != null ? apiConfig.getApiKey() : null,
                apiConfig != null ? apiConfig.getAppId() : null,
                apiConfig != null ? apiConfig.getAppSecret() : null,
                apiConfig != null ? apiConfig.getModelId() : null,
                apiConfig != null ? apiConfig.getStatus() : null);
    }

    public record AgentScopeModelProvision(
            ChatModelBase model,
            String modelConfigFingerprint) {
        public AgentScopeModelProvision {
            Objects.requireNonNull(model, "model must not be null");
            if (modelConfigFingerprint == null || modelConfigFingerprint.isBlank()) {
                throw new IllegalArgumentException("modelConfigFingerprint must not be blank");
            }
        }
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
