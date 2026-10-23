package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.importing.MediaImportData;
import com.stonewu.fusion.service.media.importing.MediaImportPlacementService;
import com.stonewu.fusion.service.media.importing.MediaImportQueries;
import com.stonewu.fusion.service.media.importing.MediaImportService;
import io.agentscope.core.tool.ToolSuspendException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;

@Service
@RequiredArgsConstructor
public class CanvasAgentImports {
    private final CanvasAgentAccess access;
    private final MediaImportService imports;
    private final MediaImportQueries queries;
    private final MediaImportPlacementService placement;
    private final CanvasJson json;

    public Object submit(ObjectNode input, ToolExecutionContext tool) {
        var session = access.require(tool);
        var sources = Arrays.asList(
                json.read(input.path("sources").toString(), MediaImportData.RequestedSource[].class));
        var request = new MediaImportData.Submit(session.requestId(), CanvasAgentRunTools.scope(session), sources,
                null);
        return summary(imports.submit(session.scope().canvasId(), request, session.context()));
    }

    public Object get(String id, ToolExecutionContext tool) {
        var session = access.require(tool);
        return summary(bound(id, session.scope(), session.context().actor()));
    }

    public Object await(String id, ToolExecutionContext tool) {
        var session = access.require(tool);
        var result = poll(id, session.scope(), session.context().actor());
        if (result != null) {
            return result;
        }
        throw new ToolSuspendException("素材导入仍在进行；完成后继续放入当前画布。");
    }

    public Object poll(String id, CanvasAgentScope scope, CanvasActor actor) {
        var job = bound(id, scope, actor);
        return MediaImportQueries.terminal(job.status()) ? summary(job) : null;
    }

    public Object place(String id, double x, double y, ToolExecutionContext tool) {
        var session = access.require(tool);
        bound(id, session.scope(), session.context().actor());
        var placed = placement.place(id,
                new MediaImportData.Place(CanvasAgentRunTools.scope(session), new MediaImportData.Point(x, y), false),
                session.context());
        var result = new LinkedHashMap<String, Object>();
        result.put("status", "success");
        result.put("import", summary(placed.job()));
        result.put("command", placed.command());
        result.put("affectedNodes", placed.command() == null ? List.of() : placed.command().changes().upsertedNodes());
        return result;
    }

    public Object control(ObjectNode input, ToolExecutionContext tool) {
        var session = access.require(tool);
        String id = input.path("importId").asText();
        bound(id, session.scope(), session.context().actor());
        if (input.path("action").asText().equals("cancel")) {
            return summary(imports.cancel(id, CanvasAgentRunTools.scope(session), session.context()));
        }
        if (!input.hasNonNull("expectedRevision") || !input.path("indices").isArray()) {
            throw new BusinessException(400, "重试需要刚读取的导入版本和失败项编号");
        }
        var indices = Arrays.asList(json.read(input.path("indices").toString(), Integer[].class));
        return summary(imports.retry(id, new MediaImportData.Retry(CanvasAgentRunTools.scope(session),
                input.path("expectedRevision").asLong(), indices), session.context()));
    }

    private MediaImportData.View bound(String id, CanvasAgentScope scope, CanvasActor actor) {
        var job = queries.get(id, actor);
        if (!job.canvasId().equals(scope.canvasId())) {
            throw new BusinessException(403, "导入任务不属于当前画布");
        }
        return job;
    }

    private Object summary(MediaImportData.View job) {
        var result = new LinkedHashMap<String, Object>();
        result.put("status", job.status());
        result.put("importId", job.id());
        result.put("canvasId", job.canvasId());
        result.put("revision", job.revision());
        result.put("canControl", job.canControl());
        result.put("cancelRequested", job.cancelRequested());
        result.put("items", job.items().stream().map(item -> {
            var entry = new LinkedHashMap<String, Object>();
            entry.put("index", item.index());
            entry.put("title", item.title());
            entry.put("source", item.source());
            entry.put("status", item.status());
            entry.put("error", item.message());
            entry.put("nodeId", item.nodeId());
            entry.put("commandId", item.commandId());
            if (item.resource() != null) {
                entry.put("mediaId", item.resource().id());
                entry.put("mediaVersion", item.resource().version());
            }
            return entry;
        }).toList());
        return result;
    }
}
