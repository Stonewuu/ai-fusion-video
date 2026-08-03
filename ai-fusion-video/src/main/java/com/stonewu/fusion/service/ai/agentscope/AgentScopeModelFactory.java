package com.stonewu.fusion.service.ai.agentscope;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelModelFactory;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.OwnedChatModel;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import com.stonewu.fusion.service.ai.provider.AiProviderService.AgentScopeModelProvision;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public final class AgentScopeModelFactory implements AgentKernelModelFactory {

    private final AiProviderService aiProviderService;

    @Override
    public OwnedChatModel create(AgentKernelSpec spec) {
        AgentScopeModelProvision provision = aiProviderService.provisionAgentScopeModel(spec.model());
        OwnedChatModel ownedModel = OwnedChatModel.owned(provision.model());
        if (spec.key().modelConfigFingerprint().equals(provision.modelConfigFingerprint())) {
            return ownedModel;
        }
        IllegalStateException changed = new IllegalStateException(
                "AgentScope provider configuration changed while creating the kernel model");
        try {
            ownedModel.close();
        } catch (Throwable closeFailure) {
            changed.addSuppressed(closeFailure);
        }
        throw changed;
    }

    public String modelConfigFingerprint(AiModel model) {
        return aiProviderService.agentScopeModelFingerprint(model);
    }
}
