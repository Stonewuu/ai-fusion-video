package com.stonewu.fusion.service.canvas.agent;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.capability.CanvasWritebackFields;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.canvas.run.writeback.CanvasWritebackService;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Service
@RequiredArgsConstructor
public class CanvasAgentOutputBinding {
    private final CanvasAgentAccess access;
    private final CanvasAgentRunTools tools;
    private final CanvasRunService runs;
    private final CanvasRunSources sources;
    private final ResourceCommandService resources;
    private final CanvasWritebackService writeback;
    private final CanvasJson json;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasGraph", allEntries = true)
    public Object bind(String runId, String outputId, String targetNodeId, String binding,
                       int version, Integer parentVersion, ToolExecutionContext context) {
        var session = access.require(context);
        tools.requireBound(runId, session);
        var target = session.node(targetNodeId);
        if (target.resource() == null) {
            throw new BusinessException(400, "结果必须写入一个明确的业务对象");
        }
        var view = runs.view(runId, session.context().actor());
        var value = view.nodes().stream().flatMap(node -> node.outputs().all().stream())
                .filter(item -> item.id().equals(outputId))
                .findFirst().orElseThrow(() -> new BusinessException(404, "指定输出不存在、尚未发布或不可访问"));
        sources.authorize(value, session.context().actor(), true);
        var ref = target.resource();
        var field = CanvasWritebackFields.require(json.object(
                Map.of("binding", binding, "targetType", ref.type().code(), "valueType", value.type().code())));
        ResourceCommand command;
        if (value.type() == TEXT) {
            if (value.text() == null || value.text().isBlank()) {
                throw new BusinessException(400, "不能写入空的生成文本");
            }
            command = ref.type() == SCRIPT && field.path().equals("rawContent")
                    ? new ScriptSourceCommand(session.requestId(), ref.id(), version, value.text())
                    : new ResourcePatchCommand(session.requestId(), ref.type(), ref.id(), version,
                    json.object(Map.of(field.path(), value.text())), parentVersion);
        } else {
            if (value.mediaId() == null || value.mediaVersion() == null) {
                throw new BusinessException(400, "写回前必须取得登记完成的媒体输出");
            }
            command = ref.type() == STORYBOARD_EPISODE
                    ? new AdoptEpisodeMediaCommand(session.requestId(), ref.id(), version, field.path(),
                    value.mediaId())
                    : new ResourceBindMediaCommand(session.requestId(), ref.type(), ref.id(), version, parentVersion,
                    Map.of(field.path(), value.mediaId()));
        }
        var receipt = resources.execute(command, session.context());
        return Map.of("status", "success", "commandId", receipt.commandId(), "affectedResources", receipt.changes(),
                "targetNodeId", targetNodeId, "binding", field, "result", receipt.result());
    }

    public Object confirm(String runId, String nodeRunId, long revision, String hash, String action,
                          String note, boolean acceptReset, ToolExecutionContext context) {
        var session = access.require(context);
        tools.requireBound(runId, session);
        writeback.decide(runId, nodeRunId, new CanvasWritebackService.Decision(revision, hash,
                        CanvasWritebackService.Action.valueOf(action), note, acceptReset, CanvasAgentRunTools.scope(session)),
                session.context().actor(), session.context());
        return CanvasAgentRunTools.summary(runs.view(runId, session.context().actor()), 0, 20);
    }
}
