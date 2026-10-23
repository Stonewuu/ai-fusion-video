package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.StoryboardReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.storyboard.dto.StoryboardStatistics;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Service
@RequiredArgsConstructor
@Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
public class StoryboardResourceQueryService {
    private final StoryboardReadRepository boards;
    private final ResourceQueryService resources;
    private final CanvasAccessPolicy access;
    private final CanvasJson json;

    public ObjectNode project(long project, CanvasActor actor) {
        access.requireProject(project, actor);
        return boards.forProject(project).map(id -> resources.get(STORYBOARD, id, actor).data()).orElse(null);
    }

    public List<ObjectNode> children(CanvasValueType type, CanvasValueType parentType, long parent, CanvasActor actor) {
        resources.get(parentType, parent, actor);
        return boards.children(type, parentType, parent).stream().map(id -> resources.get(type, id, actor).data())
                .toList();
    }

    public StoryboardStatistics statistics(long board, CanvasActor actor) {
        resources.get(STORYBOARD, board, actor);
        return new StoryboardStatistics(boards.count(STORYBOARD_EPISODE, board), boards.count(STORYBOARD_SCENE, board),
                boards.count(SHOT, board));
    }

    public ObjectNode tree(CanvasValueType type, long id, int depth, CanvasActor actor) {
        if (depth < 0 || depth > 3) {
            throw StoryboardFields.invalid("结构展开层级必须为 0 至 3");
        }
        var result = resources.get(type, id, actor).data();
        if (depth == 0 || type == SHOT) {
            return result;
        }
        String key = switch (type) {
            case STORYBOARD -> "episodes";
            case STORYBOARD_EPISODE -> "scenes";
            case STORYBOARD_SCENE -> "items";
            default -> throw StoryboardFields.invalid("不是分镜结构");
        };
        var childType = StoryboardFields.child(type);
        var children = boards.children(childType, type, id).stream()
                .map(child -> tree(childType, child, depth - 1, actor)).toList();
        result.set(key, json.object(Map.of(key, children)).get(key));
        return result;
    }
}
