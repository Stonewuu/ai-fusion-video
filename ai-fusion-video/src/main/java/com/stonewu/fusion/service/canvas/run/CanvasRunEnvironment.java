package com.stonewu.fusion.service.canvas.run;

import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasModelGraphPolicy;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeModels;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeScopes;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeTypes;
import com.stonewu.fusion.service.canvas.capability.*;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpContracts;
import com.stonewu.fusion.service.canvas.run.media.CanvasMediaExecutor;
import com.stonewu.fusion.service.canvas.run.media.CanvasMediaRules;
import com.stonewu.fusion.service.canvas.run.text.CanvasTextModels;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.VideoModelRules;
import com.stonewu.fusion.service.generation.job.GenerationModelSnapshot;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Set;

@Component
@RequiredArgsConstructor
public class CanvasRunEnvironment {
    private final CanvasJson json;
    private final CanvasModelCatalog catalog;
    private final GenerationModelSnapshot models;
    private final CanvasTextModels textModels;
    private final CanvasAgentNodeModels agentModels;
    private final CanvasAgentNodeScopes agentScopes;
    private final CanvasMcpContracts mcpContracts;
    private final CanvasCapabilityRegistry capabilities;
    private final CanvasConfigValidator configs;
    private final CanvasModelGraphPolicy modelPolicy;
    public static final Set<CanvasNodeKind> GENERATION = Set.of(CanvasNodeKind.IMAGE_GENERATION,
            CanvasNodeKind.VIDEO_GENERATION, CanvasNodeKind.COMFYUI);
    private static final Set<CanvasNodeKind> MEDIA_SOURCES = Set.of(CanvasNodeKind.IMAGE, CanvasNodeKind.VIDEO,
            CanvasNodeKind.AUDIO, CanvasNodeKind.TEXT_FILE, CanvasNodeKind.DOCUMENT);

    public void validateForRun(CanvasGraph graph, CanvasNode node, Map<String, Integer> observedOutputs) {
        var capability = capabilities.require(node.kind());
        if (capability.configFields().containsKey("modelId") && !node.config().has("modelId")) {
            throw CanvasRunException.invalid("请为“" + node.title() + "”选择模型");
        }
        if (MEDIA_SOURCES.contains(node.kind()) && node.config().path("resourceId").asText().isBlank()) {
            throw CanvasRunException.invalid("请为“" + node.title() + "”选择或上传媒体");
        }
        if ((node.kind() == CanvasNodeKind.TEXT || node.kind() == CanvasNodeKind.STORY)
                && node.config().path("content").asText().isBlank()) {
            throw CanvasRunException.invalid("请填写“" + node.title() + "”的内容");
        }
        try {
            configs.validate(capability, node.config());
        } catch (CanvasException incomplete) {
            throw CanvasRunException.invalid("请补全“" + node.title() + "”的必填配置");
        }
        if (capability.resourceType() != null && node.resource() == null) {
            throw CanvasRunException.invalid("请为“" + node.title() + "”选择关联的项目内容");
        }
        modelPolicy.validateForRun(graph, node, observedOutputs);
    }

    public ObjectNode freeze(CanvasNode node, CanvasDocument document, CanvasActor actor) {
        if (node.kind() == CanvasNodeKind.FOREACH) {
            var body = batchBody(node);
            modelPolicy.validateConfigured(body, Map.of(node.config().path("inputPort").asText(), 1));
            return json.object(Map.of("executorVersion", 1, "operation", freeze(body, document, actor)));
        }
        var result = freeze(node, actor);
        if (CanvasAgentNodeTypes.KINDS.contains(node.kind())) {
            result.set("businessTargets",
                    json.object(Map.of("values", agentScopes.targets(document, node, actor))).path("values"));
        }
        if (node.kind() == CanvasNodeKind.MCP_TOOL) {
            result.set("mcpContract", mcpContracts.freeze(node, actor));
        }
        return result;
    }

    public ObjectNode freeze(CanvasNode node, CanvasActor actor) {
        if (node.kind() == CanvasNodeKind.FOREACH) {
            var body = batchBody(node);
            modelPolicy.validateConfigured(body, Map.of(node.config().path("inputPort").asText(), 1));
            return json.object(Map.of("executorVersion", 1, "operation", freeze(body, actor)));
        }
        CanvasMediaRules.config(node);
        var result = json.object(Map.of("executorVersion", 1));
        if (CanvasAgentNodeTypes.KINDS.contains(node.kind())) {
            result.setAll(agentModels.freeze(node, actor));
        }
        if (node.kind() == CanvasNodeKind.WRITEBACK) {
            result.set("writebackField", json.object(CanvasWritebackFields.require(node.config())));
        }
        if (node.kind() == CanvasNodeKind.TEXT_GENERATION) {
            result.setAll(textModels.freeze(node.config().path("modelId").asLong()));
        }
        if (CanvasMediaExecutor.AUDIO.contains(node.kind())) {
            long apiId = node.config().path("apiConfigId").asLong();
            catalog.requireAudio(apiId);
            result.put("apiConfigId", apiId).put("apiHash", models.fingerprint(models.requireApi(apiId)));
        }
        if (GENERATION.contains(node.kind())) {
            var kind = kind(node);
            var model = node.kind() == CanvasNodeKind.COMFYUI
                    ? models.freeze(node.config().path("modelId").longValue(), kind,
                    node.config().path("workflowVersionId").longValue())
                    : models.freeze(node.config().path("modelId").longValue(), kind);
            var option = catalog.require(model.getId());
            String expected = node.kind() == CanvasNodeKind.COMFYUI ? "comfyui" : kind.name().toLowerCase();
            if (!option.kind().equals(expected)) {
                throw CanvasRunException.invalid("模型类型与节点不一致");
            }
            result.set("model", json.object(model));
            result.set("capability", json.object(option));
            result.put("apiHash", models.fingerprint(models.requireApi(model.getApiConfigId())));
            if (node.kind() == CanvasNodeKind.COMFYUI && option.versions().stream()
                    .noneMatch(version -> version.id() == node.config().path("workflowVersionId").asLong())) {
                throw CanvasRunException.invalid("工作流版本必须已发布且属于当前模型");
            }
        }
        return result;
    }

    public static GenerationKind kind(CanvasNode node) {
        return node.kind() == CanvasNodeKind.IMAGE_GENERATION
                || (node.kind() == CanvasNodeKind.COMFYUI && node.config().path("resultType").asText().equals("image"))
                ? GenerationKind.IMAGE : GenerationKind.VIDEO;
    }

    public int images(CanvasNode node) {
        if (CanvasAgentNodeTypes.KINDS.contains(node.kind())) {
            return node.config().path("maxImages").asInt(0);
        }
        if (node.kind() == CanvasNodeKind.FOREACH) {
            return Math.multiplyExact(batchItems(node), images(batchBody(node)));
        }
        return GENERATION.contains(node.kind()) && kind(node) == GenerationKind.IMAGE ? count(node) : 0;
    }

    public int seconds(CanvasNode node, ObjectNode environment) {
        if (CanvasAgentNodeTypes.KINDS.contains(node.kind())) {
            return node.config().path("maxVideoSeconds").asInt(0);
        }
        if (node.kind() == CanvasNodeKind.FOREACH) {
            return Math.multiplyExact(batchItems(node),
                    seconds(batchBody(node), (ObjectNode) environment.path("operation")));
        }
        if (!GENERATION.contains(node.kind()) || kind(node) != GenerationKind.VIDEO) {
            return 0;
        }
        var parameters = JSONUtil.parseObj(environment.path("capability").path("parameters").toString());
        int duration = VideoModelRules.budgetSeconds(parameters,
                node.config().has("duration") ? node.config().path("duration").asInt() : null);
        return Math.multiplyExact(duration, count(node));
    }

    private int count(CanvasNode node) {
        int count = node.config().path("count").asInt(1);
        if (count < 1 || count > 32) {
            throw CanvasRunException.invalid("每个生成节点一次只能生成 1 至 32 个结果");
        }
        return count;
    }

    public CanvasNode batchBody(CanvasNode node) {
        return CanvasBatchDefinition.body(node, capabilities, configs);
    }

    public int agentRuns(CanvasNode node) {
        if (node.kind() == CanvasNodeKind.FOREACH) {
            return Math.multiplyExact(batchItems(node), agentRuns(batchBody(node)));
        }
        return CanvasAgentNodeTypes.KINDS.contains(node.kind()) ? node.config().path("maxAgentRuns").asInt(16) : 0;
    }

    public int mcpCalls(CanvasNode node) {
        if (node.kind() == CanvasNodeKind.FOREACH) {
            return Math.multiplyExact(batchItems(node), mcpCalls(batchBody(node)));
        }
        return node.kind() == CanvasNodeKind.MCP_TOOL ? 1 : 0;
    }

    public int batchItems(CanvasNode node) {
        return node.kind() == CanvasNodeKind.FOREACH ? node.config().path("maxItems").asInt() : 0;
    }

    public int extraAgentRuns(CanvasNode node) {
        if (node.kind() == CanvasNodeKind.FOREACH) {
            return Math.multiplyExact(batchItems(node), extraAgentRuns(batchBody(node)));
        }
        return Math.max(0, agentRuns(node) - 1);
    }
}
