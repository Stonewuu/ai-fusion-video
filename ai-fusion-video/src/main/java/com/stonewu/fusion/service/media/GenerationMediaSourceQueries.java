package com.stonewu.fusion.service.media;

import com.stonewu.fusion.mapper.media.GenerationMediaSourceMapper;
import com.stonewu.fusion.mapper.media.GenerationMediaSourceMapper.Item;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.service.generation.GenerationTasks;
import com.stonewu.fusion.service.generation.GenerationKind;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.model.*;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Objects;

@Component
@RequiredArgsConstructor
public class GenerationMediaSourceQueries {
    private final GenerationMediaSourceMapper media;
    private final ProjectMapper projects;
    private final GenerationTasks tasks;
    private final CanvasAccessPolicy access;
    private final CanvasDocumentRepository canvases;

    public MediaSourceSnapshot ownership(MediaSourceReference source, CanvasActor actor) {
        var row = item(source, false);
        var owner = owner(row, actor);
        return new MediaSourceSnapshot(source, row.projectId(), owner.type(), owner.id(), "", "", null);
    }

    public MediaSourceSnapshot read(MediaSourceReference source, ResourceCommandContext context, boolean lock) {
        var row = item(source, false);
        owner(row, context.actor());
        if (lock) {
            if (row.projectId() != null) {
                projects.lock(row.projectId());
            }
            tasks.lock(kind(source.type()), row.taskId());
            row = item(source, true);
        }
        var owner = owner(row, context.actor());
        if (row.version() != source.version()) {
            throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "生成条目已变化，请刷新后导入");
        }
        if (row.status() != 1 || (row.taskStatus() != 2 && row.taskStatus() != 3)) {
            throw new MediaException(409, "MEDIA_SOURCE_NOT_READY", "生成结果尚未完成");
        }
        if (context.canvasId() != null) {
            var canvas = canvases.require(context.canvasId(), false);
            access.requireEdit(canvas, context.actor());
            if (!Objects.equals(canvas.projectId(), row.projectId())) {
                throw denied();
            }
        }
        var definition = field(source);
        String uri = media.value(source.type(), definition.column(), source.id(), source.version());
        if (uri == null || uri.isBlank()) {
            throw new MediaException(422, "MEDIA_SOURCE_EMPTY", "生成条目没有该媒体结果");
        }
        return new MediaSourceSnapshot(source, row.projectId(), owner.type(), owner.id(),
                source.type().code() + "-" + source.id() + "-" + source.field(), uri, definition.kind());
    }

    private Owner owner(Item row, CanvasActor actor) {
        access.requireActorScope(actor);
        if (row.projectId() != null) {
            var project = access.requireProject(row.projectId(), actor);
            return new Owner(project.ownerType(), project.ownerId());
        }
        if (Objects.equals(row.ownerType(), 1) && Objects.equals(row.ownerId(), actor.userId())) {
            return new Owner(1, actor.userId());
        }
        if (Objects.equals(row.ownerType(), 2) && Objects.equals(row.ownerId(),
                actor.teamId()) && actor.teamId() != null) {
            return new Owner(2, actor.teamId());
        }
        throw denied();
    }

    private Item item(MediaSourceReference source, boolean lock) {
        kind(source.type());
        var item = media.item(source.type(), source.id(), lock);
        if (item == null) {
            throw new MediaException(404, "MEDIA_SOURCE_NOT_FOUND", "生成条目不存在");
        }
        return item;
    }

    private GenerationKind kind(MediaSourceType type) {
        return switch (type) {
            case IMAGE_ITEM -> GenerationKind.IMAGE;
            case VIDEO_ITEM -> GenerationKind.VIDEO;
            default -> throw MediaException.invalid("不是生成条目来源");
        };
    }

    private Field field(MediaSourceReference source) {
        Map<String, Field> fields = source.type() == MediaSourceType.IMAGE_ITEM
                ? Map.of("imageUrl", new Field("image_url", MediaKind.IMAGE), "thumbnailUrl",
                new Field("thumbnail_url", MediaKind.IMAGE))
                : Map.of("videoUrl", new Field("video_url", MediaKind.VIDEO), "coverUrl",
                new Field("cover_url", MediaKind.IMAGE),
                "firstFrameUrl", new Field("first_frame_url", MediaKind.IMAGE), "lastFrameUrl",
                new Field("last_frame_url", MediaKind.IMAGE));
        var field = fields.get(source.field());
        if (field == null) {
            throw MediaException.invalid("生成媒体字段没有登记");
        }
        return field;
    }


    private MediaException denied() {
        return new MediaException(403, "MEDIA_ACCESS_DENIED", "无权访问此生成媒体来源");
    }

    private record Owner(int type, long id) {
    }

    private record Field(String column, MediaKind kind) {
    }
}
