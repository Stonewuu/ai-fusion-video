package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.service.ai.agentscope.state.StateStoreFailureGuard;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreGuardedChatModel;
import com.stonewu.fusion.service.ai.agentscope.state.AgentScopeShutdownRecoveryBridge;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.core.tool.ToolkitConfig;
import io.agentscope.harness.agent.HarnessAgent;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

@Component
@RequiredArgsConstructor
public final class AgentScopeHarnessFactory {
    private final AgentKernelModelFactory modelFactory;
    private final AgentKernelToolRegistry toolRegistry;
    private final AgentStateStore stateStore;
    private final StateStoreFailureGuard failures;
    private final AgentScopeShutdownRecoveryBridge shutdownRecoveryBridge;

    public AgentKernelResource create(AgentKernelSpec spec) {
        Objects.requireNonNull(spec, "spec must not be null");
        OwnedChatModel ownedModel = Objects.requireNonNull(
                modelFactory.create(spec), "modelFactory returned null");
        AgentKernelToolkitResources toolResources = null;
        HarnessAgent agent = null;
        try {
            Toolkit toolkit = new Toolkit(ToolkitConfig.builder()
                    .parallel(true)
                    .build());
            toolResources = Objects.requireNonNull(
                    toolRegistry.register(spec, toolkit), "toolRegistry returned null resources");
            if (!toolkit.getToolNames().equals(spec.toolWhitelist())) {
                throw new IllegalStateException(
                        "Tool registry result does not match kernel whitelist: registered="
                                + toolkit.getToolNames() + ", expected=" + spec.toolWhitelist());
            }
            agent = HarnessAgent.builder()
                    .agentId(spec.agentDefinitionStableKey())
                    .name(spec.agentName())
                    .description(spec.description())
                    .sysPrompt(spec.systemPrompt())
                    .model(new StateStoreGuardedChatModel(ownedModel.model(), failures))
                    .stateStore(stateStore)
                    .toolkit(toolkit)
                    .middleware(shutdownRecoveryBridge)
                    .maxIters(spec.maxIters())
                    .disableFilesystemTools()
                    .disableShellTool()
                    .disableMemoryTools()
                    .disableMemoryHooks()
                    .disableSessionPersistence()
                    .disableWorkspaceContext()
                    .disableAtPathExpansion()
                    .disableSubagents()
                    .disableDynamicSubagents()
                    .disableDefaultWorkspaceSkills()
                    .disableToolsConfig()
                    .disableCompaction()
                    .disableToolResultEviction()
                    .disableDynamicSkills()
                    .skillsEnabled(false)
                    .build();
            removeUnlistedHarnessTools(agent.getToolkit(), spec.toolWhitelist());
            return new AgentKernelResource(agent, ownedModel, toolResources);
        } catch (Throwable failure) {
            Throwable accumulated = failure;
            if (agent != null) {
                HarnessAgent builtAgent = agent;
                accumulated = AgentKernelResource.closeAndAccumulate(
                        accumulated, builtAgent::close);
            }
            if (toolResources != null) {
                accumulated = AgentKernelResource.closeAndAccumulate(accumulated, toolResources::close);
            }
            accumulated = AgentKernelResource.closeAndAccumulate(accumulated, ownedModel::close);
            AgentKernelResource.rethrow(accumulated);
            throw new AssertionError("unreachable");
        }
    }

    private void removeUnlistedHarnessTools(Toolkit toolkit, Set<String> whitelist) {
        Set<String> builtIns = new HashSet<>(toolkit.getToolNames());
        builtIns.removeAll(whitelist);
        builtIns.forEach(toolkit::removeTool);
        if (!toolkit.getToolNames().equals(whitelist)) {
            throw new IllegalStateException(
                    "Harness toolkit does not match kernel whitelist after built-in removal: actual="
                            + toolkit.getToolNames() + ", expected=" + whitelist);
        }
    }
}
