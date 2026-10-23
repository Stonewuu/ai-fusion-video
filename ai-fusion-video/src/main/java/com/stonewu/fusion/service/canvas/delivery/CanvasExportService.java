package com.stonewu.fusion.service.canvas.delivery;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;

import static com.stonewu.fusion.service.canvas.delivery.CanvasExportData.*;

@Service
@RequiredArgsConstructor
public class CanvasExportService {
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasExportRepository jobs;
    private final CanvasExportSources sources;
    private final CanvasJson json;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Preview preview(String canvas, Options options, CanvasActor actor) {
        return sources.preview(sources.freeze(canvas, options, actor), options.format());
    }

    @Transactional(isolation = Isolation.REPEATABLE_READ)
    @CacheEvict(value = "canvasExport", key = "#canvas")
    public Summary submit(String canvas, Submit request, CanvasActor actor) {
        access.requireView(documents.require(canvas, true), actor);
        var existing = jobs.replay(canvas, actor, request.requestId(), json.hash(request));
        if (existing != null) {
            sources.authorize(canvas, existing.manifest(), actor);
            return summary(existing, actor);
        }
        var manifest = sources.freeze(canvas, request.options(), actor);
        if (!sources.preview(manifest, request.options().format()).hash().equals(request.previewHash())) {
            throw conflict("导出内容在预览后发生变化，请重新确认范围");
        }
        return summary(jobs.insert(canvas, null, request.requestId(), json.hash(request), request.options().format(),
                manifest, ResourceCommandContext.user(actor, canvas)), actor);
    }

    @Transactional(propagation = Propagation.MANDATORY)
    @CacheEvict(value = "canvasExport", key = "#run.canvasId()")
    public String submitNode(Run run, NodeRun node) {
        var document = documents.require(run.canvasId(), true);
        access.requireExecute(document, run.actor());
        Format format = Format.valueOf(node.plan().node().config().path("format").asText().toUpperCase(Locale.ROOT));
        var included = new HashSet<String>();
        var pending = new ArrayDeque<String>();
        pending.add(node.nodeId());
        while (!pending.isEmpty()) {
            String next = pending.removeFirst();
            if (included.add(next)) {
                run.plan().nodes().stream().filter(plan -> plan.node().id().equals(next))
                        .findFirst().ifPresent(plan -> pending.addAll(plan.dependencies()));
            }
        }
        var original = run.plan().graph();
        var graph = new CanvasGraph(original.schemaVersion(), original.projectId(),
                original.nodes().stream().filter(value -> included.contains(value.id())).toList(),
                original.edges().stream().filter(edge -> included.contains(edge.sourceNodeId()) && included.contains(
                        edge.targetNodeId())).toList());
        var manifest = sources.freeze(document, graph, run.plan().documentRevision(), node.inputs().all(), format,
                run.actor());
        return jobs.insert(run.canvasId(), node.id(), "node:" + node.id(), json.hash(node.inputs()), format, manifest,
                ResourceCommandContext.user(run.actor(), run.canvasId())).id();
    }

    public Summary get(String canvas, String id, CanvasActor actor) {
        return summary(authorized(canvas, id, actor), actor);
    }

    public Job authorized(String canvas, String id, CanvasActor actor) {
        access.requireView(documents.require(canvas, false), actor);
        var job = jobs.get(id, false);
        if (!job.canvasId().equals(canvas)) {
            throw new BusinessException(404, "交付记录不存在");
        }
        sources.authorize(canvas, job.manifest(), actor);
        return job;
    }

    public List<Summary> list(String canvas, int page, CanvasActor actor) {
        if (page < 0 || page > 10000) {
            throw invalid("交付历史分页无效");
        }
        access.requireView(documents.require(canvas, false), actor);
        return jobs.list(canvas, page).stream().map(job -> {
            try {
                sources.authorize(canvas, job.manifest(), actor);
                return summary(job, actor);
            } catch (BusinessException denied) {
                if (List.of(403, 404, 409, 410).contains(denied.getCode())) {
                    var full = summary(job, actor);
                    return new Summary(full.id(), canvas, full.format(), full.status(), "受限交付内容", null, null,
                            full.revision(), full.attempts(), "部分来源已删除或失去访问权限", full.createdAt(),
                            full.updatedAt(),
                            full.actorName(), full.canCancel(), 0, 0, true);
                }
                throw denied;
            }
        }).toList();
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasExport", key = "#canvas")
    public void cancel(String canvas, String id, CanvasActor actor) {
        var permission = access.requireView(documents.require(canvas, true), actor);
        var job = jobs.get(id, true);
        if (!job.canvasId().equals(canvas)) {
            throw new BusinessException(404, "交付记录不存在");
        }
        if (!permission.canManage() && !job.context().actor().equals(actor)) {
            throw new BusinessException(403, "只有发起人或画布管理者可以取消交付");
        }
        jobs.cancel(job, actor.userId());
    }

    private Summary summary(Job job, CanvasActor actor) {
        var permission = access.requireView(documents.require(job.canvasId(), false), actor);
        return new Summary(job.id(), job.canvasId(), job.format(),
                job.cancelRequested() && !job.terminal() ? "CANCEL_REQUESTED" : job.status(),
                job.filename(), job.byteSize(), job.contentHash(), job.revision(), job.attempts(), job.error(),
                job.createdAt(), job.updatedAt(),
                jobs.actorName(job.context().actor().userId()),
                !job.terminal() && (permission.canManage() || job.context().actor().equals(actor)),
                job.manifest().media().size(),
                (int) job.manifest().values().stream().filter(value -> value.resource() != null).count(), false);
    }
}
