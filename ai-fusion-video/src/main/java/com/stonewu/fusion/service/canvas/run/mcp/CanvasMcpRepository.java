package com.stonewu.fusion.service.canvas.run.mcp;

import com.stonewu.fusion.mapper.canvas.CanvasMcpJobMapper;
import com.stonewu.fusion.entity.canvas.CanvasMcpJobEntity;
import com.stonewu.fusion.mapper.execution.ExecutionCoordinationMapper;

import java.time.LocalDateTime;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Repository;

import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Optional;

import static com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.*;

@Repository
@RequiredArgsConstructor
public class CanvasMcpRepository {
    private final CanvasMcpJobMapper mcp;
    private final ExecutionCoordinationMapper coordination;
    private final CanvasJson json;

    public Job get(String id, boolean lock) {
        var row = lock ? mcp.lock(id) : mcp.selectById(id);
        if (row == null) {
            throw new CanvasRunException(404, "MCP_JOB_NOT_FOUND", "MCP 执行记录不存在");
        }
        return read(row);
    }

    public void insert(String id, Run run, NodeRun node, Input input) {
        mcp.insertJob(id, node.id(), run.id(), run.actor().userId(), input.contract().id(), json.write(input),
                run.deadlineAt().atOffset(ZoneOffset.UTC).toLocalDateTime());
    }

    public Optional<Job> claim(String owner) {
        coordination.lockGenerationDispatch();
        var ids = mcp.next();
        if (ids.isEmpty()) {
            return Optional.empty();
        }
        String id = ids.getFirst();
        mcp.claim(owner, id);
        return Optional.of(get(id, false));
    }

    public Job leased(Claim claim) {
        var rows = mcp.leased(claim.id(), claim.owner(), claim.epoch());
        if (rows == null) {
            throw CanvasRunException.fenced();
        }
        return read(rows);
    }

    public boolean heartbeat(Claim claim) {
        return mcp.heartbeat(claim.id(), claim.owner(), claim.epoch()) == 1;
    }

    public void begin(Claim claim) {
        if (mcp.begin(claim.owner(), claim.epoch(), claim.id()) != 1) {
            throw CanvasRunException.conflict("原 MCP 请求已登记，不能再次提交");
        }
    }

    public void received(Claim claim, ObjectNode result) {
        var prior = get(claim.id(), true);
        String hash = json.hash(result);
        if (prior.result() != null) {
            if (!hash.equals(json.hash(prior.result()))) {
                throw CanvasRunException.conflict("同一 MCP 请求返回了不一致的响应");
            }
            return;
        }
        if (mcp.received(json.write(result), hash, claim.id(), claim.owner(), claim.epoch()) != 1) {
            throw CanvasRunException.fenced();
        }
    }

    public void finish(Job job, String state, String message, boolean permit) {
        mcp.finish(state, message, permit, job.id());
    }

    public void cancel(String id) {
        mcp.cancel(id);
    }

    private Job read(CanvasMcpJobEntity r) {
        String result = r.getResultJson();
        return new Job(r.getId(), r.getNodeRunId(), r.getRunId(), r.getUserId(),
                r.getContractId(),
                json.read(r.getInputJson(), Input.class), r.getStatus(), time(r.getRequestStartedAt()),
                result == null ? null : json.read(result, ObjectNode.class), r.getCancelRequested(),
                r.getPermitHeld(),
                r.getOwnerInstanceId(), r.getOwnerEpoch(), time(r.getLeaseUntil()),
                r.getRevision(), r.getErrorMsg(), time(r.getDeadlineAt()));
    }

    private Instant time(LocalDateTime value) {
        return value == null ? null : value.toInstant(ZoneOffset.UTC);
    }
}
