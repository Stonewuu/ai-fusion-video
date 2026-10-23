package com.stonewu.fusion.service.ai.configuration;

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
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpec;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelSpecFactory;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentScopeMcpRegistry;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentScopeSkillRegistry;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentUserSkillService;
import com.stonewu.fusion.service.ai.run.kernel.CanonicalAgentKernelSnapshotBuilder;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeModels;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.text.CanvasTextModels;
import com.stonewu.fusion.service.generation.job.GenerationModelSnapshot;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class AgentUserConfigurationTests {
    final ObjectMapper mapper = new ObjectMapper();
    final AiAgentRegistry registry = new AiAgentRegistry();
    final Map<String, String> saved = new HashMap<>();
    final AgentUserOverrideRepository repository = mock(AgentUserOverrideRepository.class);
    final AgentUserSkillService userSkills = mock(AgentUserSkillService.class);
    final AgentScopeV2Properties properties = new AgentScopeV2Properties();
    AgentDefinitionConfiguration definitions;
    AgentKernelSpecFactory factory;
    AiAgentService agents;
    AiModel model;
    AiAgentDefinition.SubAgentToolDef child;

    @BeforeEach void setup() {
        child = AiAgentDefinition.SubAgentToolDef.builder().toolName("delegate_review").displayName("复核")
                .description("复核内容").parametersSchema("{\"type\":\"object\",\"properties\":{}}")
                .refAgentType("reviewer").systemPromptOverride("父定义中的内置子提示词").build();
        registry.register(AiAgentDefinition.builder().type("writer").name("写作")
                .systemPrompt("内置写作提示词").enableTools(1).toolNames(List.of("read_script"))
                .subAgentTools(List.of(child)).build());
        registry.register(AiAgentDefinition.builder().type("reviewer").name("复核")
                .systemPrompt("内置复核提示词").enableTools(1).toolNames(List.of("read_script")).build());
        when(repository.get(anyLong(), anyString())).thenAnswer(call ->
                new AgentUserOverrideRepository.Stored(1, saved.getOrDefault(call.getArgument(0) + ":" + call.getArgument(1), "{}")));
        definitions = new AgentDefinitionConfiguration(registry, repository, mapper, properties);
        agents = new AiAgentService(registry, definitions);
        var selection = new AgentSkillSelection(mock(AgentScopeSkillRegistry.class), userSkills);
        var skills = new AgentKernelSkills(selection, agents, mapper);
        var models = mock(AgentScopeModelFactory.class);
        model = AiModel.builder().id(9L).name("text").code("text").modelProtocol("openai").status(1).config("{}").build();
        when(models.modelConfigFingerprint(model)).thenReturn("a".repeat(64));
        var tool = mock(ToolExecutor.class);
        when(tool.getToolName()).thenReturn("read_script");
        when(tool.isEnabled()).thenReturn(true);
        when(tool.getParametersSchema()).thenReturn("{\"type\":\"object\",\"properties\":{}}");
        factory = new AgentKernelSpecFactory(agents, new AiToolConfigService(List.of(tool), registry), models, properties,
                mapper, mock(AgentScopeMcpRegistry.class), skills);
        when(userSkills.list(42L)).thenReturn(List.of(skill("root-skill", "根任务规范"), skill("child-skill", "子任务规范"), skill("chosen", "本次规范")));
    }

    @Test void overridesAreIsolatedAndExplicitEmptyValuesDoNotInherit() {
        saved.put("42:writer", "{\"name\":\"我的编剧\",\"toolNames\":[],\"defaultSkills\":[],\"instructionTemplate\":\"\",\"maxIters\":0}");
        assertThat(agents.getConfigured("writer", 42L).getName()).isEqualTo("我的编剧");
        assertThat(agents.getConfigured("writer", 43L).getName()).isEqualTo("写作");
        assertThat(registry.getByType("writer").getToolNames()).containsExactly("read_script");
        var root = root(new AiChatReqVO().setAgentType("writer"));
        assertThat(root.toolWhitelist()).containsExactly("delegate_review");
        assertThat(root.maxIters()).isEqualTo(Integer.MAX_VALUE);
        saved.put("42:writer", "{\"toolsEnabled\":false}");
        assertThat(root(new AiChatReqVO().setAgentType("writer")).toolWhitelist()).isEmpty();
    }

    @Test void frozenPromptsToolsSkillsAndChildOverridesSurviveLaterSettingsChanges() {
        saved.put("42:writer", "{\"name\":\"我的编剧\",\"systemPrompt\":\"我的系统规范\",\"defaultSkills\":[\"root-skill\"]}");
        saved.put("42:reviewer", "{\"name\":\"我的复核助手\",\"systemPrompt\":\"用户复核规范\",\"toolNames\":[],\"defaultSkills\":[\"child-skill\"],\"maxIters\":13}");
        var root = root(new AiChatReqVO().setAgentType("writer"));
        assertThat(root.systemPrompt()).contains("我的系统规范", "根任务规范");
        var frozen = new CanonicalAgentKernelSnapshotBuilder().build(root);
        saved.put("42:writer", "{\"toolsEnabled\":false}");
        saved.put("42:reviewer", "{\"systemPrompt\":\"新规范\",\"maxIters\":2}");
        when(userSkills.list(42L)).thenReturn(List.of());
        var restored = factory.restore(frozen.payload(), model, "a".repeat(64));
        var childSpec = factory.createChild(restored, child, new ProjectContext(7L), Map.of());
        assertThat(restored.systemPrompt()).isEqualTo(root.systemPrompt());
        assertThat(restored.toolWhitelist()).contains("delegate_review", "read_script");
        assertThat(childSpec.systemPrompt()).contains("用户复核规范", "根任务规范", "子任务规范").doesNotContain("新规范", "父定义中的内置子提示词");
        assertThat(childSpec.maxIters()).isEqualTo(13);
        assertThat(childSpec.agentName()).isEqualTo("我的复核助手");
        assertThat(childSpec.toolWhitelist()).isEmpty();
    }

    @Test void nodeCustomSkillsReplaceDefaultsForBothRootAndChildrenAndEmptyMeansNone() {
        saved.put("42:writer", "{\"defaultSkills\":[\"root-skill\"]}");
        saved.put("42:reviewer", "{\"defaultSkills\":[\"child-skill\"]}");
        var custom = root(new AiChatReqVO().setAgentType("writer").setInheritAgentSkills(false).setEnabledSkills(List.of("chosen")));
        assertThat(custom.systemPrompt()).contains("本次规范").doesNotContain("根任务规范");
        assertThat(factory.createChild(custom, child, new ProjectContext(7L), Map.of()).systemPrompt())
                .contains("本次规范").doesNotContain("根任务规范", "子任务规范");
        var empty = root(new AiChatReqVO().setAgentType("writer").setInheritAgentSkills(false).setEnabledSkills(List.of()));
        assertThat(empty.systemPrompt()).doesNotContain("本任务启用的 Skills");
        assertThat(factory.createChild(empty, child, new ProjectContext(7L), Map.of()).systemPrompt()).doesNotContain("本任务启用的 Skills");
    }

    @Test void invalidDefaultsAreRejectedBeforeStartingIncludingInheritedChildLimit() throws Exception {
        saved.put("42:writer", "{\"defaultSkills\":[\"deleted\"]}");
        assertThatThrownBy(() -> root(new AiChatReqVO().setAgentType("writer"))).hasMessageContaining("Skill 不可用");
        var names = IntStream.range(0, 9).mapToObj(i -> "skill-" + i).toList();
        when(userSkills.list(42L)).thenReturn(names.stream().map(name -> skill(name, name)).toList());
        saved.put("42:writer", mapper.writeValueAsString(Map.of("defaultSkills", names.subList(0, 8))));
        saved.put("42:reviewer", mapper.writeValueAsString(Map.of("defaultSkills", List.of(names.getLast()))));
        assertThatThrownBy(() -> root(new AiChatReqVO().setAgentType("writer"))).hasMessageContaining("最多选择 8");
    }

    @Test void childAndSkillChangesInvalidateNewKernelIdentityButOldSnapshotDoesNotReadOverrides() {
        var first = root(new AiChatReqVO().setAgentType("writer"));
        saved.put("42:reviewer", "{\"systemPrompt\":\"新复核规范\"}");
        var next = root(new AiChatReqVO().setAgentType("writer"));
        assertThat(next.key()).isNotEqualTo(first.key());
        assertThat(agents.fromSnapshot("reviewer", Map.of()).getSystemPrompt()).isEqualTo("内置复核提示词");
        saved.put("42:writer", "{\"defaultSkills\":[\"chosen\"]}");
        var before = root(new AiChatReqVO().setAgentType("writer"));
        when(userSkills.list(42L)).thenReturn(List.of(skill("chosen", "修改过的规范")));
        assertThat(root(new AiChatReqVO().setAgentType("writer")).key()).isNotEqualTo(before.key());
    }

    @Test void httpCannotForgeInternalFrozenConfigurations() throws Exception {
        var request = mapper.readValue("""
                {"agentType":"writer","agentConfigurationSnapshot":"forged","agentSkillSnapshot":"forged"}
                """, AiChatReqVO.class);
        assertThat(request.getAgentConfigurationSnapshot()).isNull();
        assertThat(request.getAgentSkillSnapshot()).isNull();
    }

    @Test void categoriesDescribeBusinessPurposeWithoutChangingToolDefinitions() {
        assertThat(AgentToolCategories.category("get_generation_model_capabilities")).isEqualTo("媒体生成");
        assertThat(AgentToolCategories.category("get_project_script")).isEqualTo("剧本创作");
        assertThat(AgentToolCategories.category("update_storyboard_item_video")).isEqualTo("分镜创作");
        assertThat(AgentToolCategories.category("query_canvas_library")).isEqualTo("画布操作");
        assertThatThrownBy(() -> AgentToolCategories.category("unregistered_tool")).hasMessageContaining("登记展示分类");
    }

    @Test void canvasPreviewIncludesUserConfigurationAndSkillContentWhileCustomEmptyOverridesDefaults() {
        var text = mock(CanvasTextModels.class);
        when(text.freeze(9L)).thenAnswer(call -> mapper.createObjectNode().put("modelId", 9));
        var skills = new AgentKernelSkills(new AgentSkillSelection(mock(AgentScopeSkillRegistry.class), userSkills), agents, mapper);
        var models = new CanvasAgentNodeModels(text, mock(GenerationModelSnapshot.class), mock(CanvasModelCatalog.class), agents, skills, new CanvasJson(mapper));
        var config = mapper.createObjectNode().put("agentType", "concept_visualizer").put("modelId", 9).put("inputType", "text").put("resultType", "text");
        var node = new CanvasNode("node", CanvasNodeKind.AGENT, 1, "构思", null, config, null, 0, 0);
        saved.put("42:concept_visualizer", "{\"defaultSkills\":[\"chosen\"]}");
        var first = models.freeze(node, new CanvasActor(42, null));
        assertThat(first.path(AgentKernelSkills.SNAPSHOT).asText()).contains("本次规范");
        assertThat(models.freeze(node, new CanvasActor(43, null))).isNotEqualTo(first);
        when(userSkills.list(42L)).thenReturn(List.of(skill("chosen", "更新过的正文")));
        assertThat(models.freeze(node, new CanvasActor(42, null))).isNotEqualTo(first);
        config.put("skillMode", "CUSTOM").putArray("skillNames");
        assertThat(models.freeze(node, new CanvasActor(42, null)).path(AgentKernelSkills.SNAPSHOT).asText()).isEqualTo("[]");
        config.put("skillMode", "INHERIT").withArray("skillNames").add("chosen");
        assertThatThrownBy(() -> models.freeze(node, new CanvasActor(42, null))).hasMessageContaining("单独指定");
    }

    private AgentKernelSpec root(AiChatReqVO request) {
        request.setToolExecutionMode("FULL_ACCESS");
        return factory.createRoot(request, model, agents.getConfigured(request.getAgentType(), 42L).getSystemPrompt(), 42L);
    }
    private AgentUserSkillService.UserSkill skill(String name, String content) {
        return new AgentUserSkillService.UserSkill(name, name, name, "检查规范", content, "workspace:user");
    }
}
