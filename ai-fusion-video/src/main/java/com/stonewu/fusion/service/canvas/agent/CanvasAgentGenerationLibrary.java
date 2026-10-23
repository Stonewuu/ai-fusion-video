package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationItemViews;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.GenerationTaskViews;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.video.VideoGenerationService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

@Component
@RequiredArgsConstructor
public class CanvasAgentGenerationLibrary {
    public record Page(List<Object> items, boolean hasMore) {
    }

    private final ImageGenerationService images;
    private final VideoGenerationService videos;
    private final GenerationExecutionContexts executions;
    private final GenerationTaskViews views;
    private final CanvasJson json;

    public Page query(boolean image, Long taskId, int page, CanvasAgentAccess.Session session) {
        var actor = session.context().actor();
        var project = session.scope().projectId();
        List<Object> result = new ArrayList<>();
        if (taskId == null) {
            var projects = project == null ? List.<Long>of() : List.of(project);
            List<ObjectNode> tasks;
            long total;
            if (image) {
                var values = images.pageByUser(actor, projects, page + 1, 40);
                total = values.getTotal();
                tasks = views.images(values, actor).getList().stream().map(json::object).toList();
            } else {
                var values = videos.pageByUser(actor, projects, page + 1, 40);
                total = values.getTotal();
                tasks = views.videos(values, actor).getList().stream().map(json::object).toList();
            }
            for (var task : tasks) {
                if (Objects.equals(project, task.hasNonNull("projectId") ? task.path("projectId").asLong() : null)) {
                    var summary = new LinkedHashMap<String, Object>();
                    summary.put("id", task.path("id"));
                    summary.put("taskId", task.path("taskId"));
                    summary.put("status", task.path("status"));
                    summary.put("execution", task.path("execution"));
                    summary.put("count", task.path("count"));
                    summary.put("successCount", task.path("successCount"));
                    String prompt = task.path("prompt").asText("");
                    summary.put("promptExcerpt", prompt.substring(0, Math.min(240, prompt.length())));
                    result.add(summary);
                }
            }
            return new Page(result, (long) (page + 1) * 40 < total);
        }
        executions.requireRead(image ? GenerationKind.IMAGE : GenerationKind.VIDEO, taskId, actor);
        List<ObjectNode> items;
        if (image) {
            var task = images.getById(taskId);
            requireProject(project, task.getProjectId());
            items = images.listItems(taskId).stream().map(item -> json.object(
                            GenerationItemViews.project(item, task.getExecutionContextJson() != null, task.getStatus())))
                    .toList();
        } else {
            var task = videos.getById(taskId);
            requireProject(project, task.getProjectId());
            items = videos.listItems(taskId).stream().map(item -> json.object(
                            GenerationItemViews.project(item, task.getExecutionContextJson() != null, task.getStatus())))
                    .toList();
        }
        for (var item : items.stream().skip((long) page * 40).limit(40).toList()) {
            List<Object> sources = new ArrayList<>();
            for (String field : image ? List.of("imageUrl", "thumbnailUrl") : List.of("videoUrl", "coverUrl",
                    "firstFrameUrl", "lastFrameUrl")) {
                if (item.path("status").asInt() == 1 && !item.path(field).asText("").isBlank()) {
                    sources.add(
                            Map.of("type", image ? "imageItem" : "videoItem", "id", item.path("id").asLong(), "version",
                                    item.path("version").asInt(), "field", field));
                }
            }
            result.add(Map.of("id", item.path("id"), "version", item.path("version"), "status", item.path("status"),
                    "sources", sources));
        }
        return new Page(result, (long) (page + 1) * 40 < items.size());
    }

    private void requireProject(Long expected, Long actual) {
        if (!Objects.equals(expected, actual)) {
            throw new BusinessException(403, "生成任务不属于当前画布的项目范围");
        }
    }
}
