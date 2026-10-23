package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
@RequiredArgsConstructor
public class StoryboardCommandSupport {
    private final ResourceReadRepository resources;
    private final CanvasAccessPolicy access;

    public Long authorize(CanvasValueType type, long id, ResourceCommandContext context, boolean deletedReplay) {
        var header = resources.find(type, id, false, deletedReplay).orElseThrow(() -> new CanvasException(404,
                CanvasErrorCode.RESOURCE_DELETED, "分镜对象不存在或已删除", Map.of()));
        access.requireProject(header.projectId(), context.actor());
        return header.projectId();
    }

    @CacheEvict(value = {"storyboard", "storyboardEpisode", "storyboardScene", "storyboardItem", "storyboardStatistics", "resourceDocument"}, allEntries = true)
    public void evict() {
    }
}
