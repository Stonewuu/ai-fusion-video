package com.stonewu.fusion.service.media;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.media.model.MediaSourceSnapshot;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Objects;
import java.util.Set;

@Component
@RequiredArgsConstructor
public class MediaSourceQueries {
    private final ResourceReadRepository resources;
    private final ResourceAccessPolicy resourceAccess;
    private final CanvasAccessPolicy projectAccess;
    private final CanvasDocumentRepository canvases;
    private final GenerationMediaSourceQueries generations;

    public Set<String> fields(CanvasValueType type) {
        return switch (type) {
            case PROJECT, ASSET, ASSET_ITEM, SHOT, STORYBOARD_EPISODE -> MediaSourceFields.fields(type).keySet();
            default -> Set.of();
        };
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ, noRollbackFor = BusinessException.class)
    public MediaSourceSnapshot resolve(MediaSourceReference source, ResourceCommandContext context) {
        return read(source, context, false);
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public MediaSourceSnapshot lock(MediaSourceReference source, ResourceCommandContext context) {
        return read(source, context, true);
    }

    /**
     * 历史媒体仍按当前业务访问权读取，但不要求业务对象停留在旧的正文版本。
     */
    public MediaSourceSnapshot ownership(MediaSourceReference source, CanvasActor actor) {
        validate(source);
        if (source.type().generation()) {
            return generations.ownership(source, actor);
        }
        ResourceHeader header = resources.require(source.type().businessType(), source.id(), false);
        resourceAccess.require(header, actor);
        return owner(source, header, actor, "", "", null);
    }

    /**
     * 仅供服务端受保护的运行/交付快照使用，不能接收浏览器或模板传来的任意正文。
     */
    public MediaSourceSnapshot frozen(MediaSourceReference source, ObjectNode data, CanvasActor actor) {
        validate(source);
        if (source.type().generation()) {
            throw MediaException.invalid("历史业务快照不能使用生成条目类型");
        }
        var type = source.type().businessType();
        var header = resources.historical(type, source.id());
        resourceAccess.requireHistorical(header, actor);
        if (source.version() > header.version() || data.path("id").asLong(-1) != source.id()
                || data.path("version").asInt(-1) != source.version()) {
            throw MediaException.invalid("历史业务快照身份不一致");
        }
        var value = MediaSourceFields.get(type, data, source.field());
        String label = source.type().code() + "-" + source.id() + "-" + source.field();
        if (header.projectId() != null) {
            var owner = projectAccess.requireProject(header.projectId(), actor);
            return new MediaSourceSnapshot(source, header.projectId(), owner.ownerType(), owner.ownerId(), label,
                    value.uri(), value.kind());
        }
        var current = resources.raw(type, source.id(), false);
        var master = type == CanvasValueType.ASSET ? current
                : resources.raw(CanvasValueType.ASSET, current.path("assetId").longValue(), false);
        int ownerType = master.path("ownerType").intValue();
        long ownerId = master.path("ownerId").longValue();
        if ((ownerType != 1 && ownerType != 2) || ownerId <= 0) {
            throw MediaException.invalid("来源素材的归属不完整");
        }
        return new MediaSourceSnapshot(source, null, ownerType, ownerId, label, value.uri(), value.kind());
    }

    private MediaSourceSnapshot read(MediaSourceReference source, ResourceCommandContext context, boolean lock) {
        validate(source);
        if (source.type().generation()) {
            return generations.read(source, context, lock);
        }
        var header = resources.require(source.type().businessType(), source.id(), false);
        resourceAccess.require(header, context.actor());
        if (lock) {
            // 与业务命令共用父级锁顺序，不能从镜头/子资产反向获取项目或主资产锁。
            if (header.projectId() != null) {
                resources.require(CanvasValueType.PROJECT, header.projectId(), true);
            } else {
                long root = source.type().businessType() == CanvasValueType.ASSET ? source.id()
                        : resources.data(header).path("assetId").longValue();
                resources.require(CanvasValueType.ASSET, root, true);
            }
            header = resources.require(source.type().businessType(), source.id(), true);
            resourceAccess.require(header, context.actor());
        }
        if (header.version() != source.version()) {
            throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "媒体来源对象已改变，请刷新后重新导入");
        }
        var data = resources.data(header);
        var value = MediaSourceFields.get(source.type().businessType(), data, source.field());
        if (context.canvasId() != null) {
            var canvas = canvases.require(context.canvasId(), false);
            projectAccess.requireEdit(canvas, context.actor());
            if (!Objects.equals(canvas.projectId(), header.projectId())) {
                throw new MediaException(403, "MEDIA_SOURCE_SCOPE_MISMATCH", "来源对象不属于当前画布作用域");
            }
        }
        String label = source.type().code() + "-" + source.id() + "-" + source.field();
        return owner(source, header, context.actor(), label, value.uri(), value.kind());
    }

    private MediaSourceSnapshot owner(MediaSourceReference source, ResourceHeader header, CanvasActor actor,
                                      String name, String uri, MediaKind kind) {
        if (header.projectId() != null) {
            var owner = projectAccess.requireProject(header.projectId(), actor);
            return new MediaSourceSnapshot(source, header.projectId(), owner.ownerType(), owner.ownerId(), name, uri,
                    kind);
        }
        ResourceHeader master = source.type().businessType() == CanvasValueType.ASSET ? header
                : resources.require(CanvasValueType.ASSET, resources.data(header).path("assetId").longValue(), false);
        var data = resources.data(master);
        int type = data.path("ownerType").intValue();
        long id = data.path("ownerId").longValue();
        if ((type != 1 && type != 2) || id <= 0) {
            throw MediaException.invalid("来源素材的归属不完整");
        }
        return new MediaSourceSnapshot(source, null, type, id, name, uri, kind);
    }

    private void validate(MediaSourceReference source) {
        if (source == null || source.type() == null || source.id() <= 0
                || source.version() == null || source.version() < 0) {
            throw MediaException.invalid("媒体业务来源无效");
        }
    }
}
