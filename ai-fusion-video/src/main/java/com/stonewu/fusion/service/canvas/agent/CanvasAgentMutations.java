package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasCommandService;
import com.stonewu.fusion.service.canvas.CanvasPlacementService;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ResourceDeleteCommand;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import com.stonewu.fusion.service.resource.model.ScriptSourceCommand;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class CanvasAgentMutations {
    private final CanvasAgentAccess access;
    private final CanvasAgentOperations operations;
    private final CanvasCommandService canvases;
    private final CanvasPlacementService placements;
    private final ResourceCommandService resources;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasGraph", allEntries = true)
    public Object apply(ArrayNode values, ToolExecutionContext tool) {
        var session = access.require(tool);
        var prepared = operations.prepare(values, session);
        var receipt = canvases.edit(session.scope().canvasId(), prepared.command(), session.context());
        return Map.of("status", "success", "commandId", receipt.commandId(), "canvasRevision", receipt.revision(),
                "affectedNodes", receipt.changes().upsertedNodes(), "deletedNodeIds",
                receipt.changes().deletedNodeIds(),
                "idMappings", prepared.ids(), "receipt", receipt);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasGraph", allEntries = true)
    public Object edit(String action, String nodeId, String resourceType, String title,
                       Integer expectedVersion, Integer expectedParentVersion, ObjectNode fields,
                       double x, double y, ToolExecutionContext tool) {
        var session = access.require(tool);
        if ("create".equals(action)) {
            if (resourceType == null || title == null || title.isBlank() || fields == null) {
                throw new BusinessException(400, "新建业务节点必须提供类型、标题及字段");
            }
            var placed = placements.create(session.scope().canvasId(), new CanvasPlacementService.CreateAndPlace(
                    session.requestId(), session.id("resource-node"), title, CanvasValueType.fromCode(resourceType),
                    fields, x, y), session.context());
            return Map.of("status", "success", "commandId", placed.resource().commandId(), "canvasRevision",
                    placed.canvas().revision(),
                    "affectedNodes", placed.canvas().changes().upsertedNodes(), "affectedResources",
                    placed.resource().changes(),
                    "result", placed.resource().result(), "receipt", placed.canvas());
        }
        var node = session.node(nodeId);
        if (node.resource() == null) {
            throw new BusinessException(400, "此节点没有正式业务对象，请使用图编辑修改节点参数");
        }
        if (expectedVersion == null) {
            throw new BusinessException(400, "业务修改必须携带刚读取的资源版本");
        }
        var ref = node.resource();
        var receipt = switch (action) {
            case "patch" -> {
                if (fields == null || fields.isEmpty()) {
                    throw new BusinessException(400, "修改字段不能为空");
                }
                yield resources.execute(
                        new ResourcePatchCommand(session.requestId(), ref.type(), ref.id(), expectedVersion, fields,
                                expectedParentVersion), session.context());
            }
            case "delete" -> resources.execute(
                    new ResourceDeleteCommand(session.requestId(), ref.type(), ref.id(), expectedVersion,
                            expectedParentVersion), session.context());
            case "replaceSource" -> {
                if (ref.type() != CanvasValueType.SCRIPT || fields == null || !fields.path("rawContent").isTextual()) {
                    throw new BusinessException(400, "替换原文只接受剧本节点及 rawContent 字段");
                }
                yield resources.execute(new ScriptSourceCommand(session.requestId(), ref.id(), expectedVersion,
                        fields.path("rawContent").asText()), session.context());
            }
            default -> throw new BusinessException(400, "业务操作必须为 create、patch、delete 或 replaceSource");
        };
        return Map.of("status", "success", "commandId", receipt.commandId(), "affectedResources", receipt.changes(),
                "result", receipt.result());
    }
}
