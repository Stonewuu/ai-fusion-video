package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.service.canvas.command.CanvasGraphMutation;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.stream.Collectors;

@Component
@RequiredArgsConstructor
public class CanvasMediaGraphPolicy implements CanvasGraphMutationPolicy {
    private static final Map<CanvasNodeKind, MediaKind> KINDS = Map.of(
            CanvasNodeKind.IMAGE, MediaKind.IMAGE, CanvasNodeKind.VIDEO, MediaKind.VIDEO,
            CanvasNodeKind.AUDIO, MediaKind.AUDIO, CanvasNodeKind.TEXT_FILE, MediaKind.TEXT,
            CanvasNodeKind.DOCUMENT, MediaKind.DOCUMENT);
    private final MediaResourceQueries media;
    private final MediaAccessPolicy access;

    @Override
    public void validate(CanvasDocument document, CanvasGraph before, CanvasGraphMutation mutation, CanvasActor actor) {
        var previous = before.nodes().stream().collect(Collectors.toMap(CanvasNode::id, Function.identity()));
        for (var node : mutation.changes().upsertedNodes()) {
            var kind = KINDS.get(node.kind());
            if (kind == null) {
                continue;
            }
            if (!node.config().has("resourceId")) {
                continue;
            }
            var old = previous.get(node.id());
            String id = node.config().path("resourceId").asText();
            if (old != null && old.kind() == node.kind() && Objects.equals(old.config().path("resourceId").asText(),
                    id)) {
                continue;
            }
            if (!id.matches("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")) {
                throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "媒体节点必须选择已登记的文件");
            }
            var resource = media.get(id, actor);
            access.ready(resource, actor);
            if (resource.content().kind() != kind) {
                throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "媒体类型与节点不匹配");
            }
            if (resource.projectId() != null && !Objects.equals(resource.projectId(), document.projectId())) {
                throw new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "媒体来自其他项目，请先明确导入当前项目",
                        Map.of());
            }
        }
    }
}
