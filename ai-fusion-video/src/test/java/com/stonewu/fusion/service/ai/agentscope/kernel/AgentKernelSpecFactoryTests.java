package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.config.ai.AiAgentDefinition;
import com.stonewu.fusion.config.ai.AiAgentRegistry;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.AiToolConfigService;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeModelFactory;
import com.stonewu.fusion.service.ai.agentscope.context.ProjectContext;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentScopeMcpRegistry;
import com.stonewu.fusion.service.ai.agentscope.permission.ToolExecutionMode;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AgentKernelSpecFactoryTests {

    private static final Pattern UNRESOLVED = Pattern.compile("\\{[A-Za-z][A-Za-z0-9]*}");

    private AiAgentRegistry registry;
    private AgentKernelSpecFactory factory;
    private AiModel model;
    private AgentScopeMcpRegistry mcp;

    @BeforeEach
    void setUp() {
        registry = new AiAgentRegistry();
        AiAgentService agents = new AiAgentService(registry);
        AiToolConfigService tools = new AiToolConfigService(List.of(), registry);
        AgentScopeModelFactory models = mock(AgentScopeModelFactory.class);
        model = AiModel.builder()
                .id(9L)
                .name("test-model")
                .code("test-model")
                .modelProtocol("openai")
                .status(1)
                .config("{}")
                .build();
        when(models.modelConfigFingerprint(model)).thenReturn("a".repeat(64));
        mcp = mock(AgentScopeMcpRegistry.class);
        when(mcp.manifestsForAgent(anyString())).thenReturn(List.of());
        factory = new AgentKernelSpecFactory(
                agents,
                tools,
                models,
                new AgentScopeV2Properties(),
                new ObjectMapper(),
                mcp);
    }

    @Test
    void includesDiscoveredMcpToolsInTheImmutableRootKernel() {
        AgentKernelToolManifest manifest = new AgentKernelToolManifest(
                "search_assets",
                AgentKernelToolManifest.schemaSha256("{}"),
                true,
                false);
        when(mcp.manifestsForAgent(AgentKernelSpecFactory.DEFAULT_AGENT_KEY))
                .thenReturn(List.of(manifest));

        AgentKernelSpec spec = factory.createRoot(
                request(), model, "root prompt");

        assertThat(spec.toolWhitelist()).containsExactly("search_assets");
        assertThat(spec.toolManifest()).containsExactly(manifest);
        assertThat(spec.key().toolManifestFingerprint())
                .isEqualTo(AgentKernelKey.manifestFingerprint(List.of(manifest)));
    }

    @Test
    void userMcpKernelIsBoundToTheAuthenticatedUser() {
        AgentKernelToolManifest manifest = new AgentKernelToolManifest(
                "private_search",
                AgentKernelToolManifest.schemaSha256("{}"),
                true,
                false);
        when(mcp.manifestsForAgent(AgentKernelSpecFactory.DEFAULT_AGENT_KEY, 42L))
                .thenReturn(List.of(manifest));
        when(mcp.manifestsForAgent(AgentKernelSpecFactory.DEFAULT_AGENT_KEY, 43L))
                .thenReturn(List.of(manifest));

        AgentKernelSpec first = factory.createRoot(
                request(), model, "root prompt", 42L);
        AgentKernelSpec second = factory.createRoot(
                request(), model, "root prompt", 43L);

        assertThat(AgentKernelSpecFactory.ownerUserId(first)).isEqualTo(42L);
        assertThat(first.toolWhitelist()).containsExactly("private_search");
        assertThat(first.key()).isNotEqualTo(second.key());
    }

    @Test
    void changingUserMcpConfigurationRotatesTheKernelKey() {
        when(mcp.manifestsForAgent(AgentKernelSpecFactory.DEFAULT_AGENT_KEY, 42L))
                .thenReturn(List.of());
        when(mcp.userConfigurationFingerprint(42L))
                .thenReturn("a".repeat(64), "b".repeat(64));

        AgentKernelSpec first = factory.createRoot(
                request(), model, "root prompt", 42L);
        AgentKernelSpec second = factory.createRoot(
                request(), model, "root prompt", 42L);

        assertThat(first.key()).isNotEqualTo(second.key());
    }

    @Test
    void activeMcpReferencesFilterOnlyMcpTools() {
        AgentKernelToolManifest search = new AgentKernelToolManifest(
                "search_assets",
                AgentKernelToolManifest.schemaSha256("{}"),
                true,
                false);
        AgentKernelToolManifest inspect = new AgentKernelToolManifest(
                "inspect_media",
                AgentKernelToolManifest.schemaSha256("{}"),
                true,
                false);
        when(mcp.manifestsForAgent(AgentKernelSpecFactory.DEFAULT_AGENT_KEY))
                .thenReturn(List.of(search, inspect));

        AgentKernelSpec spec = factory.createRoot(
                request().setEnabledMcpTools(List.of("inspect_media")),
                model,
                "root prompt");

        assertThat(spec.toolWhitelist()).containsExactly("inspect_media");
        assertThat(spec.toolManifest()).containsExactly(inspect);
    }

    @Test
    void explicitEmptyMcpReferencesDisableMcpTools() {
        AgentKernelToolManifest search = new AgentKernelToolManifest(
                "search_assets",
                AgentKernelToolManifest.schemaSha256("{}"),
                true,
                false);
        when(mcp.manifestsForAgent(AgentKernelSpecFactory.DEFAULT_AGENT_KEY))
                .thenReturn(List.of(search));

        AgentKernelSpec spec = factory.createRoot(
                request().setEnabledMcpTools(List.of()),
                model,
                "root prompt");

        assertThat(spec.toolWhitelist()).isEmpty();
        assertThat(spec.toolManifest()).isEmpty();
    }

    @Test
    void activeReferenceContextUsesSupportedPromptVariableNames() {
        AiChatReqVO request = request().setContext(Map.of(
                "activeSkillReferences", "fusion-video-workflow",
                "activeMcpReferences", "assets/search_assets"));

        AgentKernelSpec spec = factory.createRoot(request, model, "root prompt");

        assertThat(spec.promptVariables())
                .containsEntry("activeSkillReferences", "fusion-video-workflow")
                .containsEntry("activeMcpReferences", "assets/search_assets");
    }

    @Test
    void inheritedRootVariablesRenderEveryConfiguredPlatformChildKernel() {
        AiChatReqVO request = request()
                .setProjectId(7L)
                .setContext(Map.of("scriptId", 41L, "storyboardId", 73L));

        assertChildrenRender(request, "script_full_parse",
                Map.of("scriptEpisodeId", 5L));
        assertChildrenRender(request, "story_to_script",
                Map.of("scriptEpisodeId", 6L));
        assertChildrenRender(request, "script_to_storyboard",
                Map.of(
                        "scriptEpisodeId", 7L,
                        "scriptEpisodeIds", "7,8"));
    }

    @Test
    void durableSnapshotRetainsVariablesNeededByFutureChildCalls() {
        AiChatReqVO request = request()
                .setAgentType("script_full_parse")
                .setProjectId(7L)
                .setContext(Map.of("scriptId", 41L));
        AgentKernelSpec root = factory.createRoot(request, model, "root prompt");

        var snapshot = new CanonicalAgentKernelSnapshotBuilder().build(root);

        assertThat(snapshot.payload().promptVariables())
                .containsEntry("projectId", "7")
                .containsEntry("scriptId", "41");
        assertThat(snapshot.snapshotJson()).contains("\"promptVariables\"");
    }

    @Test
    void capabilityPresetDoesNotReplaceMissingSnapshotProtocolIdentity() {
        AiChatReqVO request = requestForScript(41L)
                .setAgentType("script_full_parse");
        model.setModelProtocol(null);
        model.setCapabilityPresetCode("gpt-image-1");
        AgentKernelSpec spec = factory.createRoot(request, model, "root prompt");

        assertThatThrownBy(() -> new CanonicalAgentKernelSnapshotBuilder().build(spec))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("provider must not be blank");
    }

    @Test
    void scriptFullParseTemplateMatchesFrontendRequestWithoutScriptContent() {
        AiChatReqVO request = request()
                .setAgentType("script_full_parse")
                .setProjectId(7L)
                .setContext(Map.of("scriptId", 41L));
        AiAgentDefinition definition = registry.getByType("script_full_parse");
        Map<String, String> variables = AgentPromptVariables.fromRequest(request);

        String instruction = AgentPromptVariables.render(
                definition.getInstructionTemplate(), variables);
        String userMessage = AgentPromptVariables.render(
                definition.getDefaultUserMessage(), variables);

        assertThat(instruction)
                .contains("<project_id>7</project_id>")
                .contains("<script_id>41</script_id>")
                .doesNotContain("scriptContent");
        assertThat(userMessage).contains("项目 7", "ID: 41");
    }

    @Test
    void typedSubAgentPromptsDoNotRequireNaturalLanguageIdParsing() {
        for (String parentType : List.of(
                "script_full_parse", "story_to_script", "script_to_storyboard")) {
            assertThat(registry.getByType(parentType).getSystemPrompt())
                    .as(parentType)
                    .doesNotContain("message 参数", "从 message", "message 中");
        }
        for (String childType : List.of(
                "episode_scene_writer", "episode_script_creator",
                "episode_storyboard_writer")) {
            assertThat(registry.getByType(childType).getSystemPrompt())
                    .as(childType)
                    .doesNotContain("message 参数", "从 message", "message 中");
        }
    }

    @Test
    void childOnlyPromptVariablesParticipateInHarnessCacheIdentity() {
        AiChatReqVO firstRequest = requestForScript(41L);
        AiChatReqVO secondRequest = requestForScript(42L);

        AgentKernelSpec first = factory.createRoot(firstRequest, model, "same root prompt");
        AgentKernelSpec second = factory.createRoot(secondRequest, model, "same root prompt");

        assertThat(first.systemPrompt()).isEqualTo(second.systemPrompt());
        assertThat(first.key()).isNotEqualTo(second.key());
        assertThat(first.key().promptVersion()).isNotEqualTo(second.key().promptVersion());
    }

    @Test
    void platformSubAgentToolsAreDeclaredConcurrencySafe() {
        AiChatReqVO request = requestForScript(41L)
                .setAgentType("script_full_parse");

        AgentKernelSpec spec = factory.createRoot(request, model, "root prompt");

        assertThat(spec.toolManifest())
                .filteredOn(manifest -> "episode_scene_writer".equals(
                        manifest.toolName()))
                .singleElement()
                .satisfies(manifest -> {
                    assertThat(manifest.readOnly()).isFalse();
                    assertThat(manifest.concurrencySafe()).isTrue();
                });
    }

    @Test
    void restoresPersistedToolsOnlyWhenTheLiveContractStillMatches() {
        ToolExecutor tool = mock(ToolExecutor.class);
        when(tool.getToolName()).thenReturn("read_script");
        when(tool.getParametersSchema()).thenReturn("""
                {"type":"object","properties":{"scriptId":{"type":"integer"}}}
                """);
        when(tool.isEnabled()).thenReturn(true);
        when(tool.isReadOnly()).thenReturn(true);
        when(tool.isConcurrencySafe()).thenReturn(true);
        AgentScopeModelFactory models = mock(AgentScopeModelFactory.class);
        when(models.modelConfigFingerprint(model)).thenReturn("a".repeat(64));
        AgentKernelSpecFactory toolFactory = new AgentKernelSpecFactory(
                new AiAgentService(registry),
                new AiToolConfigService(List.of(tool), registry),
                models,
                new AgentScopeV2Properties(),
                new ObjectMapper(),
                mcp);
        AgentKernelSpec original = toolFactory.createRoot(
                request().setEnabledTools(List.of("read_script")),
                model,
                "root prompt");
        var payload = new CanonicalAgentKernelSnapshotBuilder().build(original).payload();

        AgentKernelSpec restored = toolFactory.restore(payload, model, "b".repeat(64));

        assertThat(restored.toolWhitelist()).containsExactly("read_script");
        assertThat(restored.toolManifest()).containsExactlyElementsOf(original.toolManifest());
        assertThat(restored.key().modelConfigFingerprint()).isEqualTo("b".repeat(64));

        when(tool.isReadOnly()).thenReturn(false);
        assertThatThrownBy(() -> toolFactory.restore(payload, model, "b".repeat(64)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("Persisted AgentScope tool is unavailable: read_script");
    }

    private void assertChildrenRender(
            AiChatReqVO request,
            String parentType,
            Map<String, Object> allInputs) {
        request.setAgentType(parentType);
        AgentKernelSpec parent = factory.createRoot(request, model, "root prompt");
        AiAgentDefinition definition = registry.getByType(parentType);
        for (AiAgentDefinition.SubAgentToolDef childDefinition : definition.getSubAgentTools()) {
            Map<String, Object> input = childDefinition.getToolName()
                    .equals("storyboard_asset_preprocessor")
                    ? Map.of("scriptEpisodeIds", allInputs.get("scriptEpisodeIds"))
                    : Map.of("scriptEpisodeId", allInputs.get("scriptEpisodeId"));
            AgentKernelSpec child = factory.createChild(
                    parent, childDefinition, new ProjectContext(7L), input);

            assertThat(UNRESOLVED.matcher(child.systemPrompt()).find())
                    .as(childDefinition.getToolName())
                    .isFalse();
            assertThat(child.systemPrompt())
                    .contains("<project_id>7</project_id>")
                    .contains("<script_id>41</script_id>")
                    .doesNotContain("message 参数", "从 message", "message 中");
            assertThat(child.promptVariables())
                    .containsEntry("scriptId", "41")
                    .containsEntry("projectId", "7");
            if (childDefinition.getToolName().equals("episode_storyboard_writer")) {
                assertThat(child.systemPrompt())
                        .contains("<storyboard_id>73</storyboard_id>")
                        .contains("<script_episode_id>7</script_episode_id>");
            }
        }
    }

    private AiChatReqVO requestForScript(long scriptId) {
        return request()
                .setAgentType("script_full_parse")
                .setProjectId(7L)
                .setContext(Map.of("scriptId", scriptId));
    }

    private AiChatReqVO request() {
        return new AiChatReqVO()
                .setToolExecutionMode(ToolExecutionMode.DEFAULT.name());
    }
}
