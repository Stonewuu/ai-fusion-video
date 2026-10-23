package com.stonewu.fusion.service.canvas.run.text;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.Optional;

import static com.stonewu.fusion.service.canvas.run.text.CanvasTextData.*;

@Service
@RequiredArgsConstructor
public class CanvasTextJobs {
    private final CanvasTextRepository tasks;
    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasRunSources sources;
    private final CanvasTextModels models;
    private final CanvasJson json;

    @Transactional(propagation = Propagation.MANDATORY)
    @CacheEvict(value = "canvasTextJob", allEntries = true)
    public String submit(Run run, NodeRun node) {
        var environment = node.plan().environment();
        var input = new Input(1, node.id(), json.read(environment.path("model").toString(), AiModel.class),
                environment.path("providerHash").asText(), node.plan().node().config().path("instruction").asText(""),
                node.inputs());
        models.verify(input);
        input.inputs().all().forEach(value -> sources.authorize(value, run.actor(), true));
        return tasks.insert(run, node, input);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<Job> claim(String owner) {
        return tasks.claim(owner);
    }

    public boolean heartbeat(Claim claim) {
        return tasks.heartbeat(claim);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Run check(Claim claim, boolean allowCancel) {
        var observed = tasks.get(claim.id(), false);
        var run = runs.run(observed.runId(), false);
        runs.lockActor(run.actor());
        var document = documents.require(run.canvasId(), true);
        access.requireView(document, run.actor());
        var current = runs.run(run.id(), true);
        var job = tasks.leased(claim);
        if (!allowCancel) {
            access.requireExecute(document, current.actor());
            if (current.cancelRequested() || job.cancelRequested() || document.archived()
                    || !job.deadlineAt().isAfter(Instant.now()) || !current.deadlineAt().isAfter(Instant.now())) {
                throw CanvasRunException.fenced();
            }
            job.input().inputs().all().forEach(value -> sources.authorize(value, current.actor(), true));
        }
        return current;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasTextJob", key = "#claim.id()")
    public void begin(Claim claim) {
        check(claim, false);
        models.verify(tasks.get(claim.id(), false).input());
        tasks.begin(claim);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasTextJob", key = "#claim.id()")
    public void received(Claim claim, ObjectNode result) {
        tasks.received(claim, result);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasTextJob", key = "#claim.id()")
    public void finish(Claim claim, String state, String message, boolean permit) {
        if (state.equals("SUCCEEDED")) {
            check(claim, false);
        }
        var job = tasks.leased(claim);
        if (state.equals("SUCCEEDED") && job.result() == null) {
            throw CanvasRunException.invalid("文本响应尚未持久化");
        }
        tasks.finish(job, state, message, permit);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasTextJob", key = "#id")
    public void cancel(String id, CanvasActor actor) {
        var job = tasks.get(id, false);
        if (!runs.run(job.runId(), false).actor().equals(actor)) {
            throw CanvasRunException.invalid("不能跨作用域取消文本任务");
        }
        tasks.cancel(id);
    }
}
