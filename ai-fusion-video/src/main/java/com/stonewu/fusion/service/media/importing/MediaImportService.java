package com.stonewu.fusion.service.media.importing;

import com.stonewu.fusion.mapper.media.MediaJobMapper;
import com.stonewu.fusion.mapper.canvas.CanvasMediaImportMapper;
import com.stonewu.fusion.mapper.canvas.CanvasMediaImportWithdrawalMapper;
import com.stonewu.fusion.entity.canvas.CanvasMediaImportWithdrawalEntity;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.MediaSourceQueries;
import com.stonewu.fusion.service.media.job.MediaJob;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

import static com.stonewu.fusion.service.media.importing.MediaImportData.*;

@Service
@RequiredArgsConstructor
public class MediaImportService {
    private final MediaJobMapper mediaJobs;
    private final CanvasMediaImportMapper imports;
    private final CanvasMediaImportWithdrawalMapper withdrawals;
    private final CanvasJson json;
    private final MediaImportAccess access;
    private final MediaImportQueries queries;
    private final MediaSourceQueries sources;
    private final ResourceAgentFence agents;
    private final MediaJobRepository jobs;
    private final GenerationOutbox outbox;

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public View submit(String canvasId, Submit request, ResourceCommandContext context) {
        request.scope().require(context.actor());
        validateRequest(request);
        if (!Objects.equals(canvasId, context.canvasId())) {
            throw MediaImportAccess.denied();
        }
        var observed = access.canvas(canvasId, context.actor(), true, false);
        agents.requireActive(context, observed.projectId());
        access.actorLock(context.actor());
        var canvas = access.canvas(canvasId, context.actor(), true, true);
        String hash = json.hash(
                Map.of("canvasId", canvasId, "request", request, "actor", context.actor(), "origin", context.origin(),
                        "agentRunId", context.agentRunId() == null ? "" : context.agentRunId(),
                        "toolCallId", context.toolCallId() == null ? "" : context.toolCallId()));
        var previous = mediaJobs.lockRequest(context.actor().userId(), request.requestId());
        if (previous != null) {
            var row = previous;
            if (!KIND.equals(row.getKind()) || !hash.equals(row.getRequestHash())) {
                throw requestConflict();
            }
            return queries.get(row.getId(), context.actor());
        }
        var withdrawn = withdrawals.get(context.actor().userId(), request.requestId(), false);
        if (withdrawn != null) {
            requireWithdrawalScope(withdrawn, canvasId, context.actor());
            throw new MediaException(410, "MEDIA_IMPORT_WITHDRAWN", "此导入请求已撤回，不会创建新的任务");
        }
        var input = new ArrayList<Source>();
        for (var requested : request.sources()) {
            input.add(new Source(requested.title().trim(), UUID.randomUUID().toString(),
                    sources.resolve(requested.source(), context)));
        }
        var checkpoint = new Checkpoint(1,
                input.stream().map(ignored -> new Item("PENDING", 1, null, null, null, null)).toList(), null);
        var worker = new ResourceCommandContext(context.actor(), "system", canvasId, context.agentRunId(),
                context.toolCallId(), null, null);
        String id = UUID.randomUUID().toString();
        mediaJobs.insertImport(id, canvas.projectId(), context.actor().userId(), request.requestId(), hash,
                json.write(worker), json.write(new Input(1, List.copyOf(input))), json.write(checkpoint));
        imports.insertImport(id, canvasId, context.actor().teamId(),
                request.position() == null ? null : json.write(new Placement(request.position(), true)));
        event(jobs.get(id), "MEDIA_IMPORT_ACCEPTED", Map.of("count", input.size(), "origin", context.origin()));
        return queries.get(id, context.actor());
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public View cancel(String id, GenerationClientScope scope, CanvasActor actor) {
        return cancel(id, scope, ResourceCommandContext.user(actor, jobs.get(id).context().canvasId()));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public View cancel(String id, GenerationClientScope scope, ResourceCommandContext context) {
        var actor = context.actor();
        scope.require(actor);
        var job = locked(id, actor, false);
        requireControl(context, job);
        if (job.cancelRequested()) {
            return queries.get(id, actor);
        }
        var checkpoint = access.checkpoint(job);
        var items = new ArrayList<>(checkpoint.items());
        String status = job.status();
        if (status.equals("QUEUED") || status.equals("WAITING")) {
            cancelUnfinished(items);
            status = "CANCELLED";
        }
        mediaJobs.cancelImport(status, json.write(new Checkpoint(1, List.copyOf(items), checkpoint.lastRetryHash())),
                id);
        event(jobs.get(id), "MEDIA_IMPORT_CANCELLED", origin(context));
        return queries.get(id, actor);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public Withdrawal withdraw(String canvasId, String requestId, GenerationClientScope scope, CanvasActor actor) {
        scope.require(actor);
        requireId(requestId);
        access.actorLock(actor);
        access.canvas(canvasId, actor, false, true);
        var previous = withdrawals.get(actor.userId(), requestId, true);
        if (previous != null) {
            requireWithdrawalScope(previous, canvasId, actor);
        }
        var records = mediaJobs.lockRequest(actor.userId(), requestId);
        if (records != null) {
            var existing = jobs.get(records.getId());
            if (!canvasId.equals(existing.context().canvasId()) || !actor.equals(existing.context().actor())) {
                throw requestConflict();
            }
        }
        View job = records == null ? null : cancel(records.getId(), scope, actor);
        if (job != null && !job.canvasId().equals(canvasId)) {
            throw requestConflict();
        }
        if (previous == null) {
            withdrawals.insert(actor.userId(), requestId, canvasId, actor.teamId());
        }
        return new Withdrawal(job != null, job);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public View retry(String id, Retry request, CanvasActor actor) {
        return retry(id, request, ResourceCommandContext.user(actor, jobs.get(id).context().canvasId()));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "mediaJob", allEntries = true)
    public View retry(String id, Retry request, ResourceCommandContext requester) {
        var actor = requester.actor();
        request.scope().require(actor);
        var job = locked(id, actor, true);
        requireControl(requester, job);
        var checkpoint = access.checkpoint(job);
        String hash = json.hash(request);
        if (hash.equals(checkpoint.lastRetryHash())) {
            return queries.get(id, actor);
        }
        if (job.revision() != request.expectedRevision() || !MediaImportQueries.terminal(job.status())) {
            throw changed();
        }
        if (request.indices() == null || request.indices().isEmpty() || request.indices().size() > MAX_ITEMS
                || new HashSet<>(request.indices()).size() != request.indices().size()) {
            throw MediaException.invalid("请选择未完成的导入项");
        }
        var items = new ArrayList<>(checkpoint.items());
        var before = new ArrayList<Map<String, Object>>();
        var context = job.context();
        for (Integer index : request.indices()) {
            if (index == null || index < 0 || index >= items.size()) {
                throw MediaException.invalid("导入项编号无效");
            }
            var item = items.get(index);
            if (!List.of("FAILED", "CANCELLED").contains(item.status())) {
                throw changed();
            }
            var source = access.input(job).sources().get(index).snapshot();
            if (!source.equals(sources.resolve(source.reference(), context))) {
                throw new MediaException(409, "MEDIA_SOURCE_CHANGED", "来源已经变化，请按当前版本创建新的导入任务");
            }
            before.add(Map.of("index", index, "attempt", item.attempt(), "status", item.status(), "errorCode",
                    item.errorCode() == null ? "" : item.errorCode()));
            items.set(index, new Item("PENDING", item.attempt() + 1, null, null, null, null));
        }
        mediaJobs.retryImport(json.write(new Checkpoint(1, List.copyOf(items), hash)), id);
        event(jobs.get(id), "MEDIA_IMPORT_RETRY", Map.of("items", before, "actor", origin(requester)));
        return queries.get(id, actor);
    }

    private void requireControl(ResourceCommandContext context, MediaJob job) {
        if (!Objects.equals(context.canvasId(), job.context().canvasId())) {
            throw MediaImportAccess.denied();
        }
        agents.requireActive(context, job.projectId());
    }

    private Map<String, Object> origin(ResourceCommandContext context) {
        return Map.of("origin", context.origin(), "agentRunId",
                context.agentRunId() == null ? "" : context.agentRunId(),
                "toolCallId", context.toolCallId() == null ? "" : context.toolCallId());
    }

    private MediaJob locked(String id, CanvasActor actor, boolean edit) {
        var job = jobs.get(id);
        access.actorLock(actor);
        access.job(job, actor, edit, true);
        mediaJobs.lock(id);
        return jobs.get(id);
    }

    public void event(MediaJob job, String type, Object detail) {
        outbox.enqueue("media-import:" + job.id() + ":" + job.revision(), type, job.id(),
                Map.of("jobId", job.id(), "canvasId", job.context().canvasId(), "revision", job.revision(), "status",
                        job.status(), "detail", detail));
    }

    public static void cancelUnfinished(List<Item> items) {
        for (int index = 0; index < items.size(); index++) {
            var item = items.get(index);
            if (item.status().equals("PENDING") || item.status().equals("RUNNING")) {
                items.set(index,
                        new Item("CANCELLED", item.attempt(), null, "MEDIA_IMPORT_CANCELLED", "未继续导入", null));
            }
        }
    }

    private void validateRequest(Submit request) {
        requireId(request.requestId());
        access.point(request.position());
        if (request.sources() == null || request.sources().isEmpty() || request.sources().size() > MAX_ITEMS) {
            throw MediaException.invalid("每个导入批次支持 1–50 项");
        }
        if (request.position() != null) {
            access.point(new Point(request.position().x() + Math.min(2, request.sources().size() - 1) * 360,
                    request.position().y() + (request.sources().size() - 1) / 3 * 280));
        }
        var unique = new HashSet<MediaSourceReference>();
        for (var source : request.sources()) {
            if (source == null || source.source() == null || source.title() == null || source.title()
                    .isBlank() || source.title().length() > 200
                    || !unique.add(source.source())) {
                throw MediaException.invalid("导入来源、名称无效或同一来源重复");
            }
        }
    }

    private void requireId(String id) {
        if (id == null || !id.matches("[A-Za-z0-9._:-]{1,128}")) {
            throw MediaException.invalid("导入请求标识无效");
        }
    }

    private void requireWithdrawalScope(CanvasMediaImportWithdrawalEntity row, String canvasId, CanvasActor actor) {
        if (!canvasId.equals(row.getCanvasId()) || !Objects.equals(actor.teamId(), row.getActorTeamId())) {
            throw requestConflict();
        }
    }

    private static MediaException requestConflict() {
        return new MediaException(409, "MEDIA_IMPORT_REQUEST_CONFLICT", "此请求标识已用于不同的导入内容或作用域");
    }

    public static MediaException changed() {
        return new MediaException(409, "MEDIA_IMPORT_CHANGED", "导入任务已变化，请刷新后再操作");
    }
}
