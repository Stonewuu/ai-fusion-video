package com.stonewu.fusion.service.canvas.run.mcp;

import com.fasterxml.jackson.databind.node.ObjectNode;
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
import java.util.UUID;

import static com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.*;

@Service
@RequiredArgsConstructor
public class CanvasMcpJobs {
    private final CanvasMcpRepository tasks;
    private final CanvasRunRepository runs;
    private final CanvasDocumentRepository documents;
    private final CanvasAccessPolicy access;
    private final CanvasRunSources sources;
    private final CanvasMcpContracts contracts;
    private final CanvasMcpProcessor processor;
    private final CanvasJson json;

    @Transactional(propagation = Propagation.MANDATORY)
    @CacheEvict(value = "canvasMcpJob", allEntries = true)
    public String submit(Run run, NodeRun node) {
        var contract = json.read(node.plan().environment().path("mcpContract").toString(), Contract.class);
        contracts.verify(contract, run.actor().userId());
        node.inputs().all().forEach(value -> sources.authorize(value, run.actor(), true));
        String id = UUID.randomUUID().toString();
        var arguments = processor.arguments(node.plan().node(), contract, node.inputs(), id);
        tasks.insert(id, run, node, new Input(1, contract, arguments, node.inputs()));
        return id;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Optional<Job> claim(String owner) {
        return tasks.claim(owner);
    }

    public boolean heartbeat(Claim claim) {
        return tasks.heartbeat(claim);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Run check(Claim claim) {
        var observed = tasks.get(claim.id(), false);
        var run = runs.run(observed.runId(), false);
        runs.lockActor(run.actor());
        var document = documents.require(run.canvasId(), true);
        access.requireExecute(document, run.actor());
        var current = runs.run(run.id(), true);
        var job = tasks.leased(claim);
        if (job.input().schemaVersion() != 1) {
            throw CanvasRunException.invalid("MCP 执行快照版本不受支持，请重新确认操作");
        }
        if (current.cancelRequested() || job.cancelled() || document.archived()
                || !job.deadlineAt().isAfter(Instant.now()) || !current.deadlineAt().isAfter(Instant.now())) {
            throw CanvasRunException.fenced();
        }
        contracts.verify(job.input().contract(), current.actor().userId());
        job.input().sources().all().forEach(value -> sources.authorize(value, current.actor(), true));
        return current;
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasMcpJob", key = "#claim.id()")
    public void begin(Claim claim) {
        check(claim);
        tasks.begin(claim);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasMcpJob", key = "#claim.id()")
    public void received(Claim claim, ObjectNode response) {
        tasks.received(claim, response);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasMcpJob", key = "#claim.id()")
    public void finish(Claim claim, String state, String message, boolean permit) {
        if (state.equals("SUCCEEDED")) {
            check(claim);
        }
        var job = tasks.leased(claim);
        if (state.equals("SUCCEEDED") && (job.result() == null || !job.result().path("valid").asBoolean())) {
            throw CanvasRunException.invalid("MCP 原响应尚未通过契约校验");
        }
        tasks.finish(job, state, message, permit);
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    @CacheEvict(value = "canvasMcpJob", key = "#id")
    public void cancel(String id, CanvasActor actor) {
        var job = tasks.get(id, false);
        if (!runs.run(job.runId(), false).actor().equals(actor)) {
            throw CanvasRunException.invalid("不能跨账号或团队取消 MCP 调用");
        }
        tasks.cancel(id);
    }
}
