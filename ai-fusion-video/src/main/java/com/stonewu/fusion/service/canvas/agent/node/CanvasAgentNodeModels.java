package com.stonewu.fusion.service.canvas.agent.node;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.service.ai.AiAgentService;
import com.stonewu.fusion.service.ai.configuration.AgentDefinitionConfiguration;
import com.stonewu.fusion.service.ai.configuration.AgentKernelSkills;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.text.CanvasTextModels;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.job.GenerationModelSnapshot;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

@Component
@RequiredArgsConstructor
public class CanvasAgentNodeModels {
    private final CanvasTextModels textModels;
    private final GenerationModelSnapshot generationModels;
    private final CanvasModelCatalog catalog;
    private final AiAgentService agents;
    private final AgentKernelSkills skills;
    private final CanvasJson json;

    public ObjectNode freeze(CanvasNode node, CanvasActor actor) {
        CanvasAgentNodeContracts.configuration(node);
        String type = CanvasAgentNodeTypes.type(node);
        String configuration = agents.snapshot(type, actor.userId());
        Map<String, String> variables = new LinkedHashMap<>();
        variables.put(AgentDefinitionConfiguration.SNAPSHOT_VARIABLE, configuration);
        var definition = agents.fromSnapshot(type, variables);
        if (definition == null) {
            throw CanvasRunException.invalid("专业智能体定义不可用");
        }
        var result = textModels.freeze(node.config().path("modelId").asLong());
        int maximum = node.config().path("maxAgentRuns").asInt(16);
        if (maximum < 1 || maximum > 64) {
            throw CanvasRunException.invalid("每个专业节点最多允许 1 至 64 个智能体任务（含主任务）");
        }
        result.put("maxAgentRuns", maximum);
        result.put("agentType", type).put("definitionHash", definitionHash(type, variables, new HashSet<>()));
        result.put(AgentDefinitionConfiguration.SNAPSHOT_VARIABLE, configuration);
        skills.root(definition, CanvasAgentNodeSkills.configure(new AiChatReqVO(), node), variables, actor.userId());
        result.put(AgentKernelSkills.SNAPSHOT, variables.get(AgentKernelSkills.SNAPSHOT));
        if (!Integer.valueOf(1).equals(definition.getEnableTools())
                && (node.kind() != CanvasNodeKind.AGENT || !node.config().path("resultType").asText().equals("text"))) {
            throw CanvasRunException.invalid(
                    "当前智能体关闭了工具，不能写入项目或生成媒体；请开启工具，或使用返回文本的智能体节点");
        }
        Set<String> tools = CanvasAgentNodeTypes.tools(child -> agents.fromSnapshot(child, variables), definition);
        generation(node, result, tools, GenerationKind.IMAGE, "imageModelId", "maxImages", 1000);
        generation(node, result, tools, GenerationKind.VIDEO, "videoModelId", "maxVideoSeconds", 36000);
        return result;
    }

    private void generation(CanvasNode node, ObjectNode environment, Set<String> tools, GenerationKind kind,
                            String field, String budgetField, int maximum) {
        if (!tools.contains(kind == GenerationKind.IMAGE ? "generate_image" : "generate_video")) {
            if (node.config().path(budgetField).asInt(0) > 0 || node.config().hasNonNull(field)) {
                throw CanvasRunException.invalid("当前专业智能体没有启用该生成能力，请移除无关的生成模型和预算");
            }
            return;
        }
        int budget = node.config().path(budgetField).asInt(-1);
        if (budget < 0 || budget > maximum) {
            throw CanvasRunException.invalid("请明确专业智能体的 " + budgetField + " 上限（可设为 0 禁止生成）");
        }
        if (budget == 0) {
            return;
        }
        long id = node.config().path(field).asLong();
        var model = generationModels.freeze(id, kind);
        if (!catalog.require(id).kind().equals(kind.name().toLowerCase())) {
            throw CanvasRunException.invalid("专业智能体的生成模型类型不匹配");
        }
        environment.set(field, json.object(model));
        environment.put(field + "ApiHash",
                generationModels.fingerprint(generationModels.requireApi(model.getApiConfigId())));
    }

    private String definitionHash(String type, Map<String, String> variables, Set<String> visited) {
        if (!visited.add(type)) {
            throw CanvasRunException.invalid("专业智能体的子任务定义存在循环");
        }
        var definition = agents.fromSnapshot(type, variables);
        Map<String, String> children = new LinkedHashMap<>();
        if (Integer.valueOf(1).equals(definition.getEnableTools()) && definition.getSubAgentTools() != null) {
            for (var child : definition.getSubAgentTools()) {
                children.put(child.getToolName(),
                        definitionHash(child.getRefAgentType(), variables, new HashSet<>(visited)));
            }
        }
        return json.hash(Map.of("definition", definition, "children", children));
    }
}
