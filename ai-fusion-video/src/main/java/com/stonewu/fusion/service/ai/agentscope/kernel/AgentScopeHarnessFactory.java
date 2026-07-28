package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.service.ai.agentscope.state.AgentScopeShutdownRecoveryBridge;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreFailureGuard;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreGuardedChatModel;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentScopeSkillRegistry;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceBaseStore;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.tool.Toolkit;
import io.agentscope.core.tool.ToolkitConfig;
import io.agentscope.harness.agent.HarnessAgent;
import io.agentscope.harness.agent.IsolationScope;
import io.agentscope.harness.agent.filesystem.remote.store.BaseStore;
import io.agentscope.harness.agent.filesystem.spec.RemoteFilesystemSpec;
import io.agentscope.harness.agent.memory.compaction.CompactionConfig;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

@Component
public final class AgentScopeHarnessFactory {
    private static final int COMPACTION_TRIGGER_PERCENT = 80;

    private final AgentKernelModelFactory modelFactory;
    private final AgentKernelToolRegistry toolRegistry;
    private final AgentStateStore stateStore;
    private final StateStoreFailureGuard failures;
    private final AgentScopeShutdownRecoveryBridge shutdownRecoveryBridge;
    private final AgentScopeSkillRegistry skillRegistry;
    private final BaseStore workspaceStore;

    @Autowired
    public AgentScopeHarnessFactory(
            AgentKernelModelFactory modelFactory,
            AgentKernelToolRegistry toolRegistry,
            AgentStateStore stateStore,
            StateStoreFailureGuard failures,
            AgentScopeShutdownRecoveryBridge shutdownRecoveryBridge,
            ObjectProvider<AgentScopeSkillRegistry> skillRegistries,
            ObjectProvider<AgentWorkspaceBaseStore> workspaceStores) {
        this(
                modelFactory,
                toolRegistry,
                stateStore,
                failures,
                shutdownRecoveryBridge,
                skillRegistries.getIfAvailable(AgentScopeHarnessFactory::disabledSkillRegistry),
                workspaceStores.getIfAvailable());
    }

    AgentScopeHarnessFactory(
            AgentKernelModelFactory modelFactory,
            AgentKernelToolRegistry toolRegistry,
            AgentStateStore stateStore,
            StateStoreFailureGuard failures,
            AgentScopeShutdownRecoveryBridge shutdownRecoveryBridge,
            AgentScopeSkillRegistry skillRegistry) {
        this(modelFactory, toolRegistry, stateStore, failures, shutdownRecoveryBridge,
                skillRegistry, null);
    }

    AgentScopeHarnessFactory(
            AgentKernelModelFactory modelFactory,
            AgentKernelToolRegistry toolRegistry,
            AgentStateStore stateStore,
            StateStoreFailureGuard failures,
            AgentScopeShutdownRecoveryBridge shutdownRecoveryBridge,
            AgentScopeSkillRegistry skillRegistry,
            BaseStore workspaceStore) {
        this.modelFactory = Objects.requireNonNull(modelFactory, "modelFactory must not be null");
        this.toolRegistry = Objects.requireNonNull(toolRegistry, "toolRegistry must not be null");
        this.stateStore = Objects.requireNonNull(stateStore, "stateStore must not be null");
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
        this.shutdownRecoveryBridge = Objects.requireNonNull(
                shutdownRecoveryBridge, "shutdownRecoveryBridge must not be null");
        this.skillRegistry = Objects.requireNonNull(
                skillRegistry, "skillRegistry must not be null");
        this.workspaceStore = workspaceStore;
    }

    public AgentScopeHarnessFactory(
            AgentKernelModelFactory modelFactory,
            AgentKernelToolRegistry toolRegistry,
            AgentStateStore stateStore,
            StateStoreFailureGuard failures,
            AgentScopeShutdownRecoveryBridge shutdownRecoveryBridge) {
        this(
                modelFactory,
                toolRegistry,
                stateStore,
                failures,
                shutdownRecoveryBridge,
                disabledSkillRegistry(),
                null);
    }

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
            Integer contextWindow = resolveContextWindow(
                    spec.model().getContextWindow(), ownedModel.model().getContextWindowSize());
            HarnessAgent.Builder builder = HarnessAgent.builder()
                    .agentId(spec.agentDefinitionStableKey())
                    .name(spec.agentName())
                    .description(spec.description())
                    .sysPrompt(spec.systemPrompt())
                    .model(new StateStoreGuardedChatModel(
                            ownedModel.model(), failures, contextWindow))
                    .stateStore(stateStore)
                    .toolkit(toolkit)
                    .middleware(shutdownRecoveryBridge)
                    .compaction(compactionConfig(contextWindow))
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
                    .disableToolsConfig();
            if (workspaceStore != null) {
                builder.filesystem(new RemoteFilesystemSpec(workspaceStore)
                        .isolationScope(IsolationScope.USER));
            } else {
                builder.disableDefaultWorkspaceSkills();
            }
            if (skillRegistry.enabled()) {
                builder.skillRepositories(skillRegistry.repositories());
            }
            if (workspaceStore != null || skillRegistry.enabled()) {
                builder.skillsEnabled(true);
            } else {
                builder.disableDynamicSkills()
                        .skillsEnabled(false);
            }
            agent = builder.build();
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

    static CompactionConfig compactionConfig(Integer contextWindow) {
        if (contextWindow == null || contextWindow <= 0) {
            return CompactionConfig.builder().build();
        }
        int triggerTokens = Math.max(
                1,
                (int) ((long) contextWindow * COMPACTION_TRIGGER_PERCENT / 100));
        return CompactionConfig.builder()
                .triggerMessages(0)
                .triggerTokens(triggerTokens)
                .build();
    }

    static Integer resolveContextWindow(Integer configuredContextWindow, int modelContextWindow) {
        if (configuredContextWindow != null && configuredContextWindow > 0) {
            return configuredContextWindow;
        }
        return modelContextWindow > 0 ? modelContextWindow : null;
    }

    private static AgentScopeSkillRegistry disabledSkillRegistry() {
        return new AgentScopeSkillRegistry(new AgentScopeV2Properties());
    }
}
