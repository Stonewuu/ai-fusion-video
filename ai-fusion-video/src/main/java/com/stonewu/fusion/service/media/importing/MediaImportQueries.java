package com.stonewu.fusion.service.media.importing;

import com.stonewu.fusion.mapper.canvas.CanvasMediaImportMapper;
import com.stonewu.fusion.mapper.media.MediaJobMapper;

import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaResource;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.List;

import static com.stonewu.fusion.service.media.importing.MediaImportData.*;

@Component
@RequiredArgsConstructor
public class MediaImportQueries {
    private final CanvasMediaImportMapper imports;
    private final MediaJobMapper mediaJobs;
    private final MediaJobRepository jobs;
    private final MediaImportAccess access;
    private final MediaResourceQueries media;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View get(String id, CanvasActor actor) {
        var job = jobs.get(id);
        access.job(job, actor, false, false);
        return view(job, actor);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View request(String canvasId, String requestId, CanvasActor actor) {
        access.canvas(canvasId, actor, false, false);
        var ids = imports.request(actor.userId(), requestId, canvasId, actor.teamId());
        if (ids.isEmpty()) {
            throw new MediaException(404, "MEDIA_IMPORT_NOT_ACCEPTED", "尚未查询到此导入请求");
        }
        return get(ids.getFirst(), actor);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Page list(String canvasId, int page, CanvasActor actor) {
        if (page < 0 || page > 1000) {
            throw MediaException.invalid("导入记录页码无效");
        }
        access.canvas(canvasId, actor, false, false);
        var ids = imports.history(canvasId, actor.userId(), actor.teamId(), page * 20);
        return new Page(ids.stream().limit(20).map(id -> summary(jobs.get(id))).toList(), page, ids.size() > 20);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public List<Summary> active(String canvasId, CanvasActor actor) {
        access.canvas(canvasId, actor, false, false);
        var ids = imports.active(canvasId, actor.userId(), actor.teamId());
        return ids.stream().map(id -> summary(jobs.get(id))).toList();
    }

    public Summary summary(MediaJob job) {
        var items = access.checkpoint(job).items();
        var placement = access.placement(job.id());
        int succeeded = (int) items.stream().filter(item -> item.status().equals("SUCCEEDED")).count();
        int failed = (int) items.stream().filter(item -> item.status().equals("FAILED")).count();
        int completed = (int) items.stream().filter(item -> terminal(item.status())).count();
        int readyToPlace = (int) items.stream()
                .filter(item -> item.status().equals("SUCCEEDED") && item.commandId() == null).count();
        return new Summary(job.id(), job.status(), job.revision(), job.cancelRequested(),
                placement != null && placement.automatic(),
                job.phase(), items.size(), succeeded, failed, completed, readyToPlace,
                job.createdAt().toInstant(ZoneOffset.UTC));
    }

    private View view(MediaJob job, CanvasActor actor) {
        var input = access.input(job);
        var checkpoint = access.checkpoint(job);
        var placement = access.placement(job.id());
        boolean canControl = true;
        try {
            access.job(job, actor, true, false);
        } catch (CanvasException failure) {
            if (!List.of(403, 404, 410)
                    .contains(failure.getCode()) && failure.getErrorCode() != CanvasErrorCode.CANVAS_ARCHIVED) {
                throw failure;
            }
            canControl = false;
        }
        var items = new ArrayList<ItemView>();
        for (int index = 0; index < checkpoint.items().size(); index++) {
            var source = input.sources().get(index);
            var state = checkpoint.items().get(index);
            MediaResource resource = state.resourceId() == null ? null : media.visible(state.resourceId(), actor)
                    .orElse(null);
            boolean unavailable = state.resourceId() != null && resource == null;
            items.add(
                    new ItemView(index, source.title(), source.snapshot().reference(), source.nodeId(), state.status(),
                            state.attempt(), unavailable ? "MEDIA_IMPORT_RESULT_UNAVAILABLE" : state.errorCode(),
                            unavailable ? "此素材已删除或访问权限已变化" : state.message(), state.commandId(),
                            resource));
        }
        return new View(job.id(), job.context().canvasId(), requestId(job.id()), job.status(), job.revision(),
                job.cancelRequested(),
                placement != null && placement.automatic(), canControl, job.phase(), List.copyOf(items),
                job.createdAt().toInstant(ZoneOffset.UTC), updatedAt(job.id()));
    }

    private String requestId(String id) {
        return mediaJobs.selectById(id).getRequestId();
    }

    private Instant updatedAt(String id) {
        return mediaJobs.selectById(id).getUpdatedAt()
                .toInstant(ZoneOffset.UTC);
    }

    public static boolean terminal(String status) {
        return List.of("SUCCEEDED", "FAILED", "CANCELLED", "PARTIAL_FAILED").contains(status);
    }
}
