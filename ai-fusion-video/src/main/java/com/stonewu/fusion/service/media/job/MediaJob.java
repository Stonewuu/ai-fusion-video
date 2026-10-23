package com.stonewu.fusion.service.media.job;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;

/**
 * 内部任务快照；控制器只返回 View，不能序列化执行身份。
 */
public record MediaJob(String id, String kind, Long projectId, Long targetId, Integer targetVersion,
                       ResourceCommandContext context, ObjectNode input, ObjectNode checkpoint,
                       String status, String phase, String error, String owner, long epoch,
                       boolean cancelRequested, long revision, LocalDateTime deadline, LocalDateTime createdAt) {
    public record Claim(String id, String owner, long epoch) {
    }

    public Claim claim() {
        return new Claim(id, owner, epoch);
    }

    public boolean terminal() {
        return !status.equals("QUEUED") && !status.equals("RUNNING") && !status.equals("WAITING");
    }

    public View view() {
        return new View(id, kind, projectId, targetId, status, phase, error, checkpoint.deepCopy(), revision,
                createdAt.toInstant(ZoneOffset.UTC), cancelRequested);
    }

    public record View(String id, String kind, Long projectId, Long targetId, String status, String phase,
                       String error, ObjectNode results, long revision, Instant createdAt, boolean cancelRequested) {
    }
}
