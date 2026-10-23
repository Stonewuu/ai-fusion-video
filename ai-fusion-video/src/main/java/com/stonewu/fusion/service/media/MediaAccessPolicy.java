package com.stonewu.fusion.service.media;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Objects;

@Component
@RequiredArgsConstructor
public class MediaAccessPolicy {
    private final CanvasAccessPolicy access;
    private final CanvasDocumentRepository canvases;
    private final MediaResourceRepository media;
    private final MediaSourceQueries sources;
    private final MediaBusinessReferences references;
    private final MediaPresetReferences presets;

    public record Owner(int type, long id) {
    }

    public Owner creation(Long projectId, ResourceCommandContext context) {
        CanvasActor actor = context.actor();
        access.requireActorScope(actor);
        Owner owner;
        if (projectId != null) {
            access.requireProjectEditor(projectId, actor);
            var project = access.requireProject(projectId, actor);
            owner = new Owner(project.ownerType(), project.ownerId());
        } else {
            owner = ownScope(actor);
        }
        if (context.canvasId() != null) {
            var canvas = canvases.find(context.canvasId(), false).orElseThrow(this::denied);
            access.requireEdit(canvas, actor);
            if (!Objects.equals(canvas.projectId(), projectId)) {
                throw denied();
            }
        }
        return owner;
    }

    public Owner ownScope(CanvasActor actor) {
        access.requireActorScope(actor);
        return actor.teamId() == null ? new Owner(1, actor.userId()) : new Owner(2, actor.teamId());
    }

    public void read(MediaResource resource, CanvasActor actor) {
        access.requireActorScope(actor);
        var preview = media.previewParent(resource.id());
        if (preview.isPresent()) {
            var parent = preview.get();
            if (media.previewParent(parent.id()).isPresent()) {
                throw MediaException.invalid("预览资源不能嵌套作为预览来源");
            }
            var original = media.get(parent.id(), false).resource();
            if (original.version() != parent.version()) {
                throw new MediaException(410, "MEDIA_PREVIEW_EXPIRED", "原素材版本已变化，此预览已失效");
            }
            ready(original, actor);
        } else if (!directAccess(resource, actor) && !references.canRead(resource.id(), actor) && !presets.canRead(
                resource.id())) {
            throw denied();
        }
        if ("DELETED".equals(resource.state())) {
            throw new MediaException(410, "MEDIA_DELETED", "媒体资源已删除");
        }
    }

    public void ready(MediaResource resource, CanvasActor actor) {
        read(resource, actor);
        if (!"READY".equals(resource.state())) {
            throw new MediaException(409, "MEDIA_NOT_READY", "媒体尚未接收完成");
        }
    }

    public void edit(MediaResource resource, ResourceCommandContext context) {
        read(resource, context.actor());
        Owner owner = creation(resource.projectId(), context);
        if (resource.ownerType() != owner.type() || resource.ownerId() != owner.id()) {
            throw denied();
        }
    }

    private MediaException denied() {
        return new MediaException(403, "MEDIA_ACCESS_DENIED", "无权访问此媒体资源");
    }

    private boolean directAccess(MediaResource resource, CanvasActor actor) {
        if (resource.projectId() == null) {
            return (resource.ownerType() == 1 && resource.ownerId() == actor.userId())
                    || (resource.ownerType() == 2 && Objects.equals(actor.teamId(), resource.ownerId()))
                    || canReadSource(resource, actor);
        }
        try {
            access.requireProject(resource.projectId(), actor);
            return true;
        } catch (CanvasException failure) {
            if (failure.getCode() != 403 && failure.getCode() != 404) {
                throw failure;
            }
            return false;
        }
    }

    private boolean canReadSource(MediaResource resource, CanvasActor actor) {
        for (var source : media.sources(resource.id())) {
            try {
                var current = sources.ownership(source, actor);
                if (Objects.equals(current.projectId(),
                        resource.projectId()) && current.ownerType() == resource.ownerType()
                        && current.ownerId() == resource.ownerId()) {
                    return true;
                }
            } catch (CanvasException | MediaException denied) {
                if (denied.getCode() != 403 && denied.getCode() != 404) {
                    throw denied;
                }
            }
        }
        return false;
    }
}
