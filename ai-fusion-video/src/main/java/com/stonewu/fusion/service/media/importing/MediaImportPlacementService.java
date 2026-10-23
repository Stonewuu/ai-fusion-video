package com.stonewu.fusion.service.media.importing;

import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.canvas.CanvasMediaImportMapper;
import com.stonewu.fusion.entity.canvas.CanvasMediaImportEntity;
import com.baomidou.mybatisplus.core.toolkit.Wrappers;

import com.stonewu.fusion.service.canvas.CanvasCommandService;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.command.CanvasEditCommand;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.media.importing.MediaImportData.*;

@Service
@RequiredArgsConstructor
public class MediaImportPlacementService {
    private final MediaImportAccess access;
    private final MediaImportQueries queries;
    private final MediaImportService imports;
    private final MediaJobRepository jobs;
    private final MediaResourceQueries media;
    private final CanvasCommandService commands;
    private final CanvasJson json;
    private final MediaJobMapper mediaJobs;
    private final CanvasMediaImportMapper placements;
    private final ResourceAgentFence agents;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public Placed place(String id, Place request, CanvasActor actor) {
        return place(id, request, ResourceCommandContext.user(actor, jobs.get(id).context().canvasId()));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public Placed place(String id, Place request, ResourceCommandContext context) {
        var actor = context.actor();
        request.scope().require(actor);
        access.point(request.position());
        var observed = jobs.get(id);
        if (!observed.context().canvasId().equals(context.canvasId())) {
            throw MediaImportAccess.denied();
        }
        agents.requireActive(context, observed.projectId());
        access.actorLock(actor);
        access.job(observed, actor, true, true);
        mediaJobs.lock(id);
        var job = jobs.get(id);
        var placement = access.placement(id);
        if (request.automatic() && (job.cancelRequested() || placement == null || !placement.automatic())) {
            throw new MediaException(409, "MEDIA_IMPORT_AUTO_PLACE_STOPPED",
                    "此任务已停止自动放入画布，可手动放入已完成的素材");
        }
        if (placement == null) {
            if (request.position() == null) {
                throw MediaException.invalid("请选择素材放入画布的位置");
            }
            placement = new Placement(request.position(), false);
            placements.update(Wrappers.<CanvasMediaImportEntity>lambdaUpdate().eq(CanvasMediaImportEntity::getJobId, id)
                    .set(CanvasMediaImportEntity::getPlacementJson, json.write(placement)));
        } else if (request.position() != null && !placement.point().equals(request.position())) {
            throw new MediaException(409, "MEDIA_IMPORT_POSITION_FIXED", "此批次已有落点，请在画布中移动导入后的节点");
        }
        var checkpoint = access.checkpoint(job);
        var items = new ArrayList<>(checkpoint.items());
        var input = access.input(job);
        var indices = new ArrayList<Integer>();
        var operations = new ArrayList<CanvasGraphOperation>();
        for (int index = 0; index < items.size(); index++) {
            var item = items.get(index);
            if (!item.status().equals("SUCCEEDED") || item.commandId() != null) {
                continue;
            }
            var resource = media.get(item.resourceId(), actor);
            if (!resource.state().equals("READY")) {
                throw MediaException.invalid("此项素材尚未就绪");
            }
            var source = input.sources().get(index);
            var point = new Point(placement.point().x() + index % 3 * 360, placement.point().y() + index / 3 * 280);
            access.point(point);
            var config = json.object(Map.of("resourceId", resource.id()));
            CanvasNodeKind kind = switch (resource.content().kind()) {
                case IMAGE -> CanvasNodeKind.IMAGE;
                case VIDEO -> CanvasNodeKind.VIDEO;
                case AUDIO -> CanvasNodeKind.AUDIO;
                case TEXT -> CanvasNodeKind.TEXT_FILE;
                case DOCUMENT -> CanvasNodeKind.DOCUMENT;
            };
            operations.add(new CanvasGraphOperation.CreateNode(new CanvasNode(source.nodeId(), kind, 1, source.title(),
                    new CanvasNodeLayout(point.x(), point.y(), 300.0, null, null, false, false), config, null, 0, 0)));
            indices.add(index);
        }
        if (operations.isEmpty()) {
            return new Placed(queries.get(id, actor), null);
        }
        var command = commands.edit(job.context().canvasId(),
                new CanvasEditCommand("media-import:" + id + ":" + json.hash(indices), operations), context);
        for (int index : indices) {
            var item = items.get(index);
            items.set(index,
                    new Item(item.status(), item.attempt(), item.resourceId(), item.errorCode(), item.message(),
                            command.commandId()));
        }
        mediaJobs.importPlacement(json.write(new Checkpoint(1, List.copyOf(items), checkpoint.lastRetryHash())), id);
        imports.event(jobs.get(id), "MEDIA_IMPORT_PLACED",
                Map.of("indices", indices, "commandId", command.commandId()));
        return new Placed(queries.get(id, actor), command);
    }
}
