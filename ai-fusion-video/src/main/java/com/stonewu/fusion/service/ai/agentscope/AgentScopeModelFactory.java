package com.stonewu.fusion.service.ai.agentscope;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelModelFactory;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.OwnedChatModel;
import com.stonewu.fusion.service.ai.agentscope.message.MediaResolvingChatModel;
import com.stonewu.fusion.service.ai.agentscope.message.TrackedAgentChatModel;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import com.stonewu.fusion.service.ai.provider.AiProviderService.AgentScopeModelProvision;
import com.stonewu.fusion.service.ai.run.model.AgentModelRequests;
import com.stonewu.fusion.service.media.AgentMediaInputResolver;
import io.agentscope.core.model.ChatModelBase;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public final class AgentScopeModelFactory implements AgentKernelModelFactory {

    private final AiProviderService aiProviderService;
    private final AgentMediaInputResolver media;
    private final AgentModelRequests requests;

    @Override
    public OwnedChatModel create(AgentKernelSpec spec) {
        return OwnedChatModel.owned(new MediaResolvingChatModel(
                new TrackedAgentChatModel(provisionRequest(spec), () -> provisionRequest(spec),
                        spec.model(), requests, spec.key().modelConfigFingerprint()), spec.model(), media));
    }

    private ChatModelBase provisionRequest(AgentKernelSpec spec) {
        AgentScopeModelProvision provision = aiProviderService.provisionSingleRequestModel(spec.model());
        if (spec.key().modelConfigFingerprint().equals(provision.modelConfigFingerprint())) {
            return provision.model();
        }
        IllegalStateException changed = new IllegalStateException(
                "AgentScope provider configuration changed while creating the kernel model");
        try {
            OwnedChatModel.owned(provision.model()).close();
        } catch (Throwable closeFailure) {
            changed.addSuppressed(closeFailure);
        }
        throw changed;
    }

    public String modelConfigFingerprint(AiModel model) {
        return aiProviderService.agentScopeModelFingerprint(model);
    }
}
