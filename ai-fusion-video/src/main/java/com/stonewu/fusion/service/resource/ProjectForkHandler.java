package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkSources;
import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkWriter;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class ProjectForkHandler implements ResourceCommandHandler {
    private final CanvasDocumentRepository canvases;
    private final CanvasAccessPolicy access;
    private final CanvasProjectForkSources sources;
    private final CanvasProjectForkWriter canvasWriter;
    private final ProjectContentCopies copies;
    private final CanvasJson json;

    @Override
    public CanvasValueType resourceType() {
        return CanvasValueType.PROJECT;
    }

    @Override
    public ResourceOperation operation() {
        return ResourceOperation.FORK_PROJECT;
    }

    @Override
    public Long authorize(ResourceCommand command, ResourceCommandContext context, ResourceCommandReceipt previous) {
        var input = (ProjectForkCommand) command;
        if (context.canvasId() != null || !"user".equals(context.origin())) {
            throw new BusinessException(400, "项目副本必须通过明确的项目复制操作创建");
        }
        access.requireActorScope(context.actor());
        if (previous != null) {
            access.requireProject(previous.result().path("id").longValue(), context.actor());
            access.requireView(canvases.require(previous.result().path("canvasId").asText(), false), context.actor());
        } else {
            access.requireView(canvases.require(input.sourceCanvasId(), false), context.actor());
        }
        return null;
    }

    @Override
    public ResourceMutation apply(ResourceCommand command, ResourceCommandContext context) {
        var input = (ProjectForkCommand) command;
        if (canvases.find(input.canvasId(), true).isPresent()) {
            throw new BusinessException(409, "副本画布编号已被使用");
        }
        var source = sources.read(input.sourceCanvasId(), input.expectedRevision(), context.actor(), true);
        if (!sources.preview(source).digest().equals(input.digest())) {
            throw new BusinessException(409, "复制来源在预览后发生变化，请重新确认");
        }
        if (!source.warnings().isEmpty()) {
            throw new BusinessException(409, "部分旧媒体尚未登记，请先导入素材库再创建独立副本");
        }
        Map<String, MediaResource> originals = new LinkedHashMap<>();
        source.media().forEach(file -> originals.put(file.id(), file));
        var copied = copies.create(input.title(), input.canvasId(), source.manifest().graph().projectId(),
                source.primaryScript(), source.primaryStoryboard(),
                source.objects(), originals, context.actor());
        long project = copied.projectId();
        var canvas = canvasWriter.write(input, source, project, copied.remap(), context.actor());
        var result = json.object(
                Map.of("id", project, "canvasId", canvas.id(), "title", input.title(), "sourceCanvasId",
                        input.sourceCanvasId(),
                        "sourceRevision", input.expectedRevision(), "businessObjects", copied.changes().size(),
                        "mediaFiles", copied.media().size()));
        var lineage = copied.objects().entrySet().stream()
                .map(entry -> Map.of("type", entry.getKey().type(), "sourceId", entry.getKey().id(), "targetId",
                        entry.getValue())).toList();
        var mediaLineage = copied.media().entrySet().stream()
                .map(entry -> Map.of("sourceId", entry.getKey(), "targetId", entry.getValue().id())).toList();
        return new ResourceMutation(result, copied.changes(), json.object(Map.of("created", copied.changes(),
                "fork",
                Map.of("sourceCanvasId", input.sourceCanvasId(), "sourceRevision", input.expectedRevision(), "objects",
                        lineage, "media", mediaLineage))));
    }

    @Override
    @CacheEvict(value = {"project", "script", "storyboard", "asset", "resourceDocument", "canvasGraph"}, allEntries = true)
    public void evict() {
    }
}
