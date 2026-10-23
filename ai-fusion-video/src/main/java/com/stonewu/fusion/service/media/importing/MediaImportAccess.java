package com.stonewu.fusion.service.media.importing;

import com.stonewu.fusion.mapper.canvas.CanvasMediaImportMapper;
import com.stonewu.fusion.mapper.system.UserMapper;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;

import java.util.Objects;

@Component
@RequiredArgsConstructor
public class MediaImportAccess {
    private final CanvasDocumentRepository canvases;
    private final CanvasAccessPolicy access;
    private final CanvasMediaImportMapper imports;
    private final UserMapper users;
    private final CanvasJson json;

    public CanvasDocument canvas(String id, CanvasActor actor, boolean edit, boolean lock) {
        var canvas = canvases.require(id, lock);
        if (edit) {
            access.requireEdit(canvas, actor);
        } else {
            access.requireView(canvas, actor);
        }
        return canvas;
    }

    public void job(MediaJob job, CanvasActor actor, boolean edit, boolean lock) {
        if (!MediaImportData.KIND.equals(job.kind()) || job.context().canvasId() == null
                || !job.context().actor().equals(actor)) {
            throw denied();
        }
        var document = canvas(job.context().canvasId(), actor, edit, lock);
        if (!Objects.equals(document.projectId(), job.projectId())) {
            throw denied();
        }
    }

    public MediaImportData.Input input(MediaJob job) {
        var input = json.read(json.write(job.input()), MediaImportData.Input.class);
        if (input.schemaVersion() != 1 || input.sources() == null || input.sources().isEmpty()
                || input.sources().size() > MediaImportData.MAX_ITEMS) {
            throw MediaException.invalid("导入任务的输入协议无效");
        }
        return input;
    }

    public MediaImportData.Checkpoint checkpoint(MediaJob job) {
        var checkpoint = json.read(json.write(job.checkpoint()), MediaImportData.Checkpoint.class);
        if (checkpoint.schemaVersion() != 1 || checkpoint.items() == null || checkpoint.items().size() != input(
                job).sources().size()) {
            throw MediaException.invalid("导入任务的进度协议无效");
        }
        return checkpoint;
    }

    public MediaImportData.Placement placement(String id) {
        String value = imports.selectById(id).getPlacementJson();
        return value == null ? null : json.read(value, MediaImportData.Placement.class);
    }

    public void point(MediaImportData.Point point) {
        if (point != null && (!Double.isFinite(point.x()) || !Double.isFinite(point.y())
                || Math.abs(point.x()) > 1_000_000 || Math.abs(point.y()) > 1_000_000)) {
            throw MediaException.invalid("画布落点无效");
        }
    }

    public void actorLock(CanvasActor actor) {
        var user = users.lockEnabled(actor.userId());
        if (user == null) {
            throw denied();
        }
        access.requireActorScope(actor);
    }

    public static MediaException denied() {
        return new MediaException(403, "MEDIA_IMPORT_ACCESS_DENIED", "请在原账号和团队的画布中操作此导入任务");
    }
}
