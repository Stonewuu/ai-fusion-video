package com.stonewu.fusion.service.canvas.agent;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class CanvasAgentScopeService {
    private final CanvasQueryService canvases;
    private final ResourceQueryService resources;
    private final CanvasJson json;

    public String encode(CanvasAgentScope scope) {
        return scope == null ? null : json.write(scope);
    }

    public CanvasAgentScope decode(String value) {
        return value == null ? null : json.read(value, CanvasAgentScope.class);
    }

    public CanvasAgentScope resume(String value, Long projectId, CanvasActor actor) {
        CanvasAgentScope previous = decode(value);
        if (previous == null) {
            return null;
        }
        if (previous.nodeRunId() != null) {
            throw new BusinessException(409, "专业智能体任务必须在画布中重新计划运行，不能从历史对话绕过节点预算继续");
        }
        var snapshot = canvases.snapshot(previous.canvasId(), actor);
        if (!Objects.equals(projectId, snapshot.document().projectId())) {
            throw new BusinessException(409, "历史消息的画布项目已发生变化");
        }
        var available = snapshot.graph().nodes().stream().map(CanvasNode::id).collect(Collectors.toSet());
        var focus = previous.focusedResource();
        var currentFocus = focus == null ? null : snapshot.graph().nodes().stream()
                .map(CanvasNode::resource).filter(Objects::nonNull)
                .filter(ref -> ref.type() == focus.type() && Objects.equals(ref.id(), focus.id()))
                .findFirst().orElse(null);
        if (currentFocus != null) {
            resources.get(currentFocus.type(), currentFocus.id(), actor);
        }
        return new CanvasAgentScope(previous.canvasId(), projectId,
                previous.selectedNodeIds().stream().filter(available::contains).toList(),
                previous.visibleNodeIds().stream().filter(available::contains).toList(), currentFocus,
                snapshot.lastSequence(), snapshot.lastSequence(), snapshot.document().revision());
    }

    public CanvasAgentScope capture(CanvasAgentContext request, Long projectId, CanvasActor actor) {
        if (request == null) {
            return null;
        }
        var snapshot = canvases.snapshot(request.canvasId(), actor);
        if (!Objects.equals(projectId, snapshot.document().projectId())) {
            throw new BusinessException(409, "消息项目与当前画布不一致，请重新打开画布助手");
        }
        Set<String> nodes = snapshot.graph().nodes().stream().map(CanvasNode::id).collect(Collectors.toSet());
        var selected = ids(request.selectedNodeIds(), 200, nodes, "选中节点");
        var visible = ids(request.visibleNodeIds(), 80, nodes, "可见节点");
        if (request.lastSequence() == null || request.lastSequence() < 0 || request.lastSequence() > snapshot.lastSequence()) {
            throw new BusinessException(409, "画布同步位置无效，请完成同步后重试");
        }
        var focus = request.focusedResource();
        if (focus != null) {
            boolean bound = snapshot.graph().nodes().stream().anyMatch(node -> node.resource() != null
                    && node.resource().type() == focus.type() && Objects.equals(node.resource().id(), focus.id()));
            if (!bound) {
                throw new BusinessException(403, "关注的业务对象不在当前画布中");
            }
            var resource = resources.get(focus.type(), focus.id(), actor);
            if (focus.version() != null && focus.version() != resource.header().version()) {
                throw new BusinessException(409, "关注的业务对象已更新，请保存并同步后重试");
            }
        }
        return new CanvasAgentScope(request.canvasId(), projectId, selected, visible, focus,
                request.lastSequence(), snapshot.lastSequence(), snapshot.document().revision());
    }

    public String prompt(CanvasAgentScope scope, CanvasActor actor) {
        var snapshot = canvases.snapshot(scope.canvasId(), actor);
        var ids = new HashSet<>(scope.selectedNodeIds());
        ids.addAll(scope.visibleNodeIds());
        var nodes = snapshot.graph().nodes().stream().filter(node -> ids.contains(node.id())).limit(80)
                .map(node -> Map.of("id", node.id(), "kind", node.kind().code(), "title", node.title(),
                        "configVersion", node.configVersion(), "layoutVersion", node.layoutVersion())).toList();
        return "\n\n<canvas_context>\n" + json.write(Map.of("scope", scope,
                "title", snapshot.document().title(), "permissions", snapshot.permissions(),
                "nodeCount", snapshot.graph().nodes().size(), "nodes", nodes)) + "\n</canvas_context>";
    }

    private List<String> ids(List<String> input, int maximum, Set<String> available, String name) {
        if (input == null) {
            return List.of();
        }
        if (input.size() > maximum || input.stream().anyMatch(id -> id == null || !available.contains(id))) {
            throw new BusinessException(409, name + "已变化或超过本轮上限，请重新选择");
        }
        if (new HashSet<>(input).size() != input.size()) {
            throw new BusinessException(400, name + "不能重复");
        }
        return List.copyOf(input);
    }
}
