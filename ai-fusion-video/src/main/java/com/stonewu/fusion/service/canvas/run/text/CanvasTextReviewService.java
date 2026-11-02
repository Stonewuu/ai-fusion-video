package com.stonewu.fusion.service.canvas.run.text;

import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.run.text.CanvasTextData.Job;

@Service
@RequiredArgsConstructor
public class CanvasTextReviewService {
    public record View(String id, String status, String phase, String error, long revision, String receivedText,
                       boolean completeResponse, boolean remoteCapacityHeld, boolean cancelRequested,
                       List<String> actions) {
    }

    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasTextRepository tasks;
    private final CanvasRunSources sources;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public View detail(String runId, String nodeId, CanvasActor actor) {
        var run = runs.run(runId, false);
        access.requireView(documents.require(run.canvasId(), false), actor);
        var job = task(runId, nodeId, false);
        job.input().inputs().all().forEach(value -> sources.authorize(value, actor, false));
        return new View(job.id(), job.status(), job.phase(), job.error(), job.revision(),
                job.result() == null ? null : job.result().path("text").asText(),
                job.result() != null && job.result().path("complete").asBoolean(), job.permitHeld(),
                job.cancelRequested(),
                List.of());
    }

    private Job task(String runId, String nodeId, boolean lock) {
        var node = runs.node(nodeId, false);
        if (!node.runId().equals(runId) || !"TEXT".equals(node.externalKind()) || node.externalId() == null) {
            throw CanvasRunException.invalid("此节点没有对应的文本任务");
        }
        return tasks.get(node.externalId(), lock);
    }
}
