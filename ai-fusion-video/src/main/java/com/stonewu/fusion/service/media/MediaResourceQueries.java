package com.stonewu.fusion.service.media;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.media.model.MediaSourceSnapshot;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Optional;

/**
 * 作用域和状态每次读取数据库，内容缓存不会绕过权限撤销。
 */
@Component
@RequiredArgsConstructor
public class MediaResourceQueries {
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;
    private final CanvasAccessPolicy projects;
    private final MediaMetadataCache cache;
    private final CanvasJson json;

    public record Page(List<MediaResource> items, int page, int pageSize, boolean hasMore) {
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ, noRollbackFor = BusinessException.class)
    public MediaResource get(String id, CanvasActor actor) {
        return read(id, actor);
    }

    /**
     * 任务历史保留完成事实，失去访问权限的结果不返回媒体详情，也不阻止取消剩余任务。
     */
    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Optional<MediaResource> visible(String id, CanvasActor actor) {
        try {
            return Optional.of(read(id, actor));
        } catch (BusinessException denied) {
            if (List.of(403, 404, 410).contains(denied.getCode())) {
                return Optional.empty();
            }
            throw denied;
        }
    }

    private MediaResource read(String id, CanvasActor actor) {
        var header = resources.get(id, false).resource();
        access.read(header, actor);
        return json.read(cache.get(id, header.version()), MediaResource.class);
    }

    /**
     * 运行计划只查询显式导入的映射，不下载文件、不修改原业务地址。
     */
    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ, noRollbackFor = BusinessException.class)
    public Optional<MediaResource> imported(MediaSourceSnapshot source, CanvasActor actor) {
        var stored = resources.source(source.reference(), false);
        if (stored.isEmpty()) {
            return Optional.empty();
        }
        var row = stored.get();
        access.read(row.resource(), actor);
        if (!row.requestHash().equals(MediaSourceIdentity.hash(json, source, row.resource().content()))) {
            throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "此来源版本的内容已改变，请刷新来源后重新导入");
        }
        if (!"READY".equals(row.resource().state())) {
            return Optional.empty();
        }
        return Optional.of(read(row.resource().id(), actor));
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public List<MediaSourceReference> sources(String id, CanvasActor actor) {
        access.read(resources.get(id, false).resource(), actor);
        return resources.sources(id);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Page list(Long projectId, int page, int size, CanvasActor actor) {
        if (page < 1 || page > 10000 || size < 1 || size > 100) {
            throw MediaException.invalid("分页参数无效");
        }
        var owner = access.ownScope(actor);
        if (projectId != null) {
            projects.requireProject(projectId, actor);
        }
        var found = resources.list(projectId, owner.type(), owner.id(), size + 1, (page - 1) * size);
        found.forEach(resource -> access.read(resource, actor));
        return new Page(found.size() > size ? found.subList(0, size) : found, page, size, found.size() > size);
    }
}
