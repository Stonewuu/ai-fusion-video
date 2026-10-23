package com.stonewu.fusion.service.canvas.agent;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.configuration.AgentSkillSelection;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeContracts;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeTypes;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentProfiles;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasNodeCapability;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.*;

@Service
@RequiredArgsConstructor
public class CanvasAgentQueries {
    private final CanvasAgentAccess access;
    private final CanvasCapabilityRegistry capabilities;
    private final ResourceQueryService resources;
    private final CanvasJson json;
    private final CanvasAgentProfiles profiles;
    private final AgentSkillSelection skills;

    public Object context(List<String> kinds, ToolExecutionContext tool) {
        var session = access.require(tool);
        if (kinds != null && kinds.size() > 8) {
            throw new BusinessException(400, "每次最多读取 8 种节点的完整能力");
        }
        List<CanvasNodeCapability> full = kinds == null ? List.of() : kinds.stream()
                .map(kind -> capabilities.require(CanvasNodeKind.fromCode(kind))).toList();
        var selected = session.snapshot().graph().nodes().stream()
                .filter(node -> session.scope().selectedNodeIds().contains(node.id())).map(this::summary).toList();
        var result = new LinkedHashMap<String, Object>(
                Map.of("status", "success", "scope", session.scope(), "document", session.snapshot().document(),
                        "permissions", session.snapshot().permissions(), "selectedNodes", selected,
                        "nodeCount", session.snapshot().graph().nodes().size(), "edgeCount",
                        session.snapshot().graph().edges().size(),
                        "nodeKinds", capabilities.list().stream()
                                .map(item -> Map.of("kind", item.kind().code(), "label", item.label(),
                                        "category", item.category(), "executor", item.executor())).toList(),
                        "capabilities", full,
                        "warnings", session.scope().clientSequence() < session.snapshot().lastSequence()
                                ? List.of("客户端上下文之后画布有新事件；修改前请读取节点的当前版本") : List.of()));
        if (full.stream().anyMatch(
                item -> CanvasAgentNodeTypes.KINDS.contains(item.kind()) || item.kind() == CanvasNodeKind.FOREACH)) {
            result.put("agentProfiles", profiles.list(tool.getUserId()));
            result.put("skills", skills.catalog(tool.getUserId()));
            result.put("skillSelection",
                    "skillMode=INHERIT 沿用当前用户的智能体默认 Skill；CUSTOM 使用 skillNames 指定的 Skill（空列表表示停用），作用于节点和子任务");
        }
        if (tool.getModel() != null) {
            result.put("selectedReasoningModel", Map.of(
                    "id", tool.getModel().getId(), "name", tool.getModel().getName()));
        }
        result.put("creationInputs", Map.of("context", "文字与业务内容作为内容设定，不自动使用其中的媒体",
                "media", "图片、视频和音频必须连接到对应媒体输入，且所选模型实际支持"));
        result.put("executionPolicy", "画布读取、编辑和生成自动批准，子任务继承策略");
        return result;
    }

    public Object query(String kind, String query, String adjacentTo, int offset, int limit,
                        ToolExecutionContext tool) {
        if (offset < 0 || limit < 1 || limit > 100) {
            throw new BusinessException(400, "分页范围无效，每页最多 100 个节点");
        }
        var session = access.require(tool);
        var graph = session.snapshot().graph();
        if (kind != null) {
            CanvasNodeKind.fromCode(kind);
        }
        Set<String> adjacent = new HashSet<>();
        if (adjacentTo != null) {
            session.node(adjacentTo);
            graph.edges().forEach(edge -> {
                if (edge.sourceNodeId().equals(adjacentTo)) {
                    adjacent.add(edge.targetNodeId());
                }
                if (edge.targetNodeId().equals(adjacentTo)) {
                    adjacent.add(edge.sourceNodeId());
                }
            });
        }
        var matches = graph.nodes().stream().filter(node -> kind == null || node.kind().code().equals(kind))
                .filter(node -> query == null || node.title().toLowerCase(Locale.ROOT)
                        .contains(query.toLowerCase(Locale.ROOT)))
                .filter(node -> adjacentTo == null || adjacent.contains(node.id()))
                .sorted((a, b) -> a.id().compareTo(b.id())).toList();
        var page = matches.stream().skip(offset).limit(limit).map(this::summary).toList();
        return Map.of("status", "success", "canvasId", session.scope().canvasId(), "canvasRevision",
                session.snapshot().document().revision(),
                "nodes", page, "total", matches.size(), "offset", offset, "hasMore",
                offset + page.size() < matches.size());
    }

    public Object details(String nodeId, int offset, int limit, int edgeOffset, ToolExecutionContext tool) {
        if (offset < 0 || limit < 1 || limit > 32000) {
            throw new BusinessException(400, "内容分页范围无效，单次最多 32000 字符");
        }
        if (edgeOffset < 0) {
            throw new BusinessException(400, "连线起始位置无效");
        }
        var session = access.require(tool);
        var node = session.node(nodeId);
        var result = new LinkedHashMap<String, Object>();
        result.put("status", "success");
        result.put("canvasRevision", session.snapshot().document().revision());
        result.put("node", summary(node));
        result.put("configuration", CanvasAgentLibrary.page(json.write(node.config()), offset, limit));
        result.put("capability", capabilities.require(node.kind()));
        if (CanvasAgentNodeTypes.KINDS.contains(node.kind()) && node.config().hasNonNull("agentType")) {
            result.put("agentProfile", CanvasAgentNodeContracts.require(node.config().path("agentType").asText()));
        }
        var edges = session.snapshot().graph().edges().stream()
                .filter(edge -> edge.sourceNodeId().equals(nodeId) || edge.targetNodeId().equals(nodeId)).toList();
        result.put("edges", edges.stream().skip(edgeOffset).limit(100).toList());
        result.put("edgeOffset", edgeOffset);
        result.put("totalEdges", edges.size());
        if (node.resource() != null) {
            var resource = resources.get(node.resource().type(), node.resource().id(), session.context().actor());
            String content = json.write(resource.data());
            result.put("resource", resource.header());
            result.put("content", CanvasAgentLibrary.page(content, offset, limit));
        }
        return result;
    }

    public Object summary(CanvasNode node) {
        var value = new LinkedHashMap<String, Object>();
        value.put("id", node.id());
        value.put("kind", node.kind());
        value.put("title", node.title());
        value.put("configVersion", node.configVersion());
        value.put("layoutVersion", node.layoutVersion());
        value.put("layout", node.layout());
        value.put("resource", node.resource());
        return value;
    }
}
