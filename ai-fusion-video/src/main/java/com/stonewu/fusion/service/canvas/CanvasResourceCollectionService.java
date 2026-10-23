package com.stonewu.fusion.service.canvas;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.resource.AssetReadRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.repository.resource.StoryboardReadRepository;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/**
 * 父版本和有序子结构在同一个读取快照中返回，正文缓存仍由 ResourceQueryService 管理。授权不缓存。
 */
@Service
@RequiredArgsConstructor
public class CanvasResourceCollectionService {
    private final CanvasDocumentRepository canvases;
    private final CanvasAccessPolicy access;
    private final ResourceAccessPolicy resourceAccess;
    private final ResourceQueryService resources;
    private final ResourceReadRepository scripts;
    private final StoryboardReadRepository storyboards;
    private final AssetReadRepository assets;

    public record Entry(ResourceHeader header, ObjectNode data, boolean canEdit,
                        CanvasValueType parentType, Long parentId) {
    }

    public record Collection(Entry root, List<Entry> descendants) {
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Collection get(String canvasId, CanvasValueType type, long id, int depth, CanvasActor actor) {
        if (depth < 0 || depth > 3) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "展开层级必须为 0 至 3");
        }
        var canvas = canvases.require(canvasId, false);
        boolean editable = access.requireView(canvas, actor).canEdit() && !canvas.archived();
        var root = entry(type, id, null, null, canvas, editable, actor);
        List<Entry> descendants = new ArrayList<>();
        collect(root, depth, canvas, editable, actor, descendants);
        return new Collection(root, List.copyOf(descendants));
    }

    private Entry entry(CanvasValueType type, long id, CanvasValueType parentType, Long parentId,
                        CanvasDocument canvas, boolean editable, CanvasActor actor) {
        var value = resources.get(type, id, actor);
        if (!Objects.equals(value.header().projectId(), canvas.projectId())) {
            throw new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "对象不属于当前画布项目", Map.of());
        }
        return new Entry(value.header(), value.data(), editable && resourceAccess.canEdit(value.header(), actor),
                parentType, parentId);
    }

    private void collect(Entry parent, int depth, CanvasDocument canvas, boolean editable, CanvasActor actor,
                         List<Entry> result) {
        if (depth == 0) {
            return;
        }
        var type = parent.header().resourceType();
        var childType = childType(type);
        if (childType == null) {
            return;
        }
        long id = parent.header().resourceId();
        List<Long> ids = switch (type) {
            case SCRIPT, SCRIPT_EPISODE -> scripts.scriptChildren(childType, id);
            case STORYBOARD, STORYBOARD_EPISODE, STORYBOARD_SCENE -> storyboards.children(childType, type, id);
            case ASSET -> assets.items(id);
            default -> throw new CanvasException(CanvasErrorCode.INVALID_RESOURCE_REFERENCE, "此对象没有可展开结构");
        };
        if (result.size() + ids.size() > 2000) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "此次展开超过 2000 个对象，请选择单集或场次继续");
        }
        for (long childId : ids) {
            if (result.size() >= 2000) {
                throw new CanvasException(CanvasErrorCode.INVALID_COMMAND,
                        "此次展开超过 2000 个对象，请选择单集或场次继续");
            }
            var child = entry(childType, childId, type, id, canvas, editable, actor);
            result.add(child);
            collect(child, depth - 1, canvas, editable, actor, result);
        }
    }

    private CanvasValueType childType(CanvasValueType type) {
        return switch (type) {
            case SCRIPT -> SCRIPT_EPISODE;
            case SCRIPT_EPISODE -> SCRIPT_SCENE;
            case STORYBOARD -> STORYBOARD_EPISODE;
            case STORYBOARD_EPISODE -> STORYBOARD_SCENE;
            case STORYBOARD_SCENE -> SHOT;
            case ASSET -> ASSET_ITEM;
            default -> null;
        };
    }
}
