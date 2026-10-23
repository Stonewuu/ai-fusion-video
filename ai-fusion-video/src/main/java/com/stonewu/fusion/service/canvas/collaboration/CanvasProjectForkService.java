package com.stonewu.fusion.service.canvas.collaboration;

import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ProjectForkCommand;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
public class CanvasProjectForkService {
    private final CanvasProjectForkSources sources;
    private final ResourceCommandService commands;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public CanvasProjectForkSources.Preview preview(String canvas, long revision, CanvasActor actor) {
        return sources.preview(sources.read(canvas, revision, actor, false));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasGraph", allEntries = true)
    public ResourceCommandReceipt copy(ProjectForkCommand command, CanvasActor actor) {
        return commands.execute(command, ResourceCommandContext.user(actor, null));
    }
}
