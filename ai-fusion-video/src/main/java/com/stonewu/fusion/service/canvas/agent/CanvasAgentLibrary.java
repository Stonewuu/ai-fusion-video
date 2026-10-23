package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasResourceCollectionService;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.AssetResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardResourceQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

/**
 * 返回当前画布作用域里尚未放到画布的真实资源；不把其他项目的资料装入模型上下文。
 */
@Service
@RequiredArgsConstructor
public class CanvasAgentLibrary {
    private final CanvasAgentAccess access;
    private final ResourceQueryService resources;
    private final StoryboardResourceQueryService boards;
    private final AssetResourceQueryService assets;
    private final CanvasResourceCollectionService collections;
    private final MediaResourceQueries media;
    private final CanvasJson json;
    private final CanvasAgentGenerationLibrary generations;

    public Object query(String category, String parentType, Long parentId, String keyword, int page,
                        ToolExecutionContext tool) {
        if (page < 0 || page > 9999 || keyword != null && keyword.length() > 200) {
            throw new BusinessException(400, "资源页码或搜索词无效");
        }
        var session = access.require(tool);
        var actor = session.context().actor();
        var project = session.scope().projectId();
        List<Object> items = new ArrayList<>();
        boolean more = false;
        switch (category) {
            case "project" -> {
                if (project != null && page == 0) {
                    items.add(resource(PROJECT, resources.get(PROJECT, project, actor).data()));
                    var script = resources.script(project, actor);
                    if (script != null) {
                        items.add(resource(SCRIPT, script));
                    }
                    var board = boards.project(project, actor);
                    if (board != null) {
                        items.add(resource(STORYBOARD, board));
                    }
                }
            }
            case "children" -> {
                if (parentType == null || parentId == null || parentId <= 0) {
                    throw new BusinessException(400, "展开资源需要父对象类型和编号");
                }
                var children = collections.get(session.scope().canvasId(), CanvasValueType.fromCode(parentType),
                        parentId, 1, actor).descendants();
                for (var child : children.stream().skip((long) page * 40).limit(40).toList()) {
                    items.add(resource(child.header().resourceType(), child.data()));
                }
                more = (long) (page + 1) * 40 < children.size();
            }
            case "assets" -> {
                var values = assets.page(project, null, keyword, page + 1, 40, actor);
                for (JsonNode asset : values.path("records")) {
                    Long assetProject = asset.hasNonNull("projectId") ? asset.path("projectId").asLong() : null;
                    if (Objects.equals(assetProject, project)) {
                        items.add(resource(ASSET, asset));
                    }
                }
                more = (long) (page + 1) * 40 < values.path("total").asLong();
            }
            case "media" -> {
                var values = media.list(project, page + 1, 40, actor);
                for (MediaResource value : values.items()) {
                    if (Objects.equals(value.projectId(), project)) {
                        items.add(
                                Map.of("mediaId", value.id(), "version", value.version(), "name", value.name(), "state",
                                        value.state(),
                                        "content", value.content(), "nodeKind",
                                        value.content().kind().name().equals("TEXT") ? "textFile" : value.content()
                                                .kind().name().toLowerCase(),
                                        "config", Map.of("resourceId", value.id())));
                    }
                }
                more = values.hasMore();
            }
            case "imageHistory", "videoHistory" -> {
                var values = generations.query(category.equals("imageHistory"), parentId, page, session);
                items.addAll(values.items());
                more = values.hasMore();
            }
            default -> throw new BusinessException(400, "资源目录类型无效");
        }
        return Map.of("status", "success", "canvasId", session.scope().canvasId(), "category", category,
                "page", page, "items", items, "hasMore", more);
    }

    public Object details(String type, long id, int offset, int limit, ToolExecutionContext tool) {
        var session = access.require(tool);
        var resource = resources.get(CanvasValueType.fromCode(type), id, session.context().actor());
        if (!Objects.equals(resource.header().projectId(), session.scope().projectId())) {
            throw new BusinessException(403, "资源不属于当前画布的项目范围");
        }
        return Map.of("status", "success", "resource", resource.header(), "content",
                page(json.write(resource.data()), offset, limit));
    }

    private Object resource(CanvasValueType type, JsonNode data) {
        String title = data.hasNonNull("title") ? data.path("title").asText() : data.path("name").asText();
        return Map.of("nodeKind", type.code(), "title", title, "resource",
                Map.of("type", type, "id", data.path("id").asLong(),
                        "version", data.path("version").asLong()));
    }

    public static Object page(String content, int offset, int limit) {
        if (offset < 0 || limit < 1 || limit > 32000) {
            throw new BusinessException(400, "正文页码无效，单次最多 32000 字符");
        }
        return Map.of("format", "json", "offset", offset, "totalCharacters", content.length(),
                "text", content.substring(Math.min(offset, content.length()),
                        (int) Math.min(content.length(), (long) offset + limit)),
                "hasMore", (long) offset + limit < content.length());
    }
}
