package com.stonewu.fusion.service.execution;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.fasterxml.jackson.databind.node.TextNode;
import com.stonewu.fusion.entity.execution.ExecutionRecoveryDecisionEntity;
import com.stonewu.fusion.mapper.execution.ExecutionRecordMapper;
import com.stonewu.fusion.mapper.execution.ExecutionRecoveryDecisionMapper;
import com.stonewu.fusion.service.ai.run.AgentEventEnvelopeSanitizer;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Kind;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Ref;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Remote;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData.Target;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryTargets;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Set;

@Repository
@RequiredArgsConstructor
public class ExecutionRecordQueries {
    private static final int PAGE_SIZE = 20;
    private static final Set<String> ACTIVE = Set.of("READY", "QUEUED", "RUNNING", "WAITING", "NEEDS_REVIEW");
    private final ExecutionRecordMapper records;
    private final ExecutionRecoveryTargets targets;
    private final ExecutionRecoveryDecisionMapper decisions;
    private final AgentEventEnvelopeSanitizer sanitizer;
    private final ExecutionRecordGenerationDetails generationDetails;

    public enum Filter {
        ALL, REVIEW, FAILED, CANCELLED;

        public List<String> statuses() {
            return switch (this) {
                case ALL -> List.of("UNKNOWN_SUBMISSION", "NEEDS_REVIEW", "FAILED", "PARTIAL_FAILED", "CONFLICT", "CANCELLED");
                case REVIEW -> List.of("UNKNOWN_SUBMISSION", "NEEDS_REVIEW");
                case FAILED -> List.of("FAILED", "PARTIAL_FAILED", "CONFLICT");
                case CANCELLED -> List.of("CANCELLED");
            };
        }
    }

    public record Summary(Kind kind, String id, String title, String status, boolean localFinished,
                          boolean remoteUnknown, String message, Instant createdAt, Instant updatedAt) {
    }

    public record Audit(long adminUserId, String note, Instant createdAt) {
    }

    public record Detail(Summary task, Long projectId, String canvasId, boolean capacityHeld,
                         List<Remote> remote, List<Audit> history, ExecutionRecordGenerationDetails.View generation) {
    }

    public record Page(List<Summary> items, boolean hasMore, int page, long total, long matchedTotal) {
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Page list(long userId, int page, Filter filter, int days) {
        if (page < 0 || page > 1000000) {
            throw MediaException.invalid("任务记录页码无效");
        }
        if (days != 0 && days != 7 && days != 30) {
            throw MediaException.invalid("请选择全部时间、近 7 天或近 30 天");
        }
        LocalDateTime since = days == 0 ? null : LocalDateTime.now(ZoneOffset.UTC).minusDays(days);
        long total = records.count(userId, Filter.ALL.statuses(), null);
        long matched = filter == Filter.ALL && since == null ? total : records.count(userId, filter.statuses(), since);
        var rows = records.list(userId, filter.statuses(), since, page * PAGE_SIZE, PAGE_SIZE + 1);
        return new Page(rows.stream().limit(PAGE_SIZE).map(row -> summary(row,
                targets.get(new Ref(row.kind(), row.id())))).toList(), rows.size() > PAGE_SIZE, page, total, matched);
    }

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Detail get(long userId, Kind kind, String id) {
        var row = records.get(userId, kind, id);
        if (row == null) {
            throw new MediaException(404, "EXECUTION_RECORD_NOT_FOUND", "任务记录不存在或不可访问");
        }
        var target = targets.get(new Ref(kind, id));
        var history = decisions.selectList(Wrappers.<ExecutionRecoveryDecisionEntity>lambdaQuery()
                        .eq(ExecutionRecoveryDecisionEntity::getExecutionKind, kind.name())
                        .eq(ExecutionRecoveryDecisionEntity::getJobId, id)
                        .orderByAsc(ExecutionRecoveryDecisionEntity::getCreatedAt)).stream()
                .map(value -> new Audit(value.getAdminUserId(), safeMessage(value.getNote()),
                        value.getCreatedAt().toInstant(ZoneOffset.UTC))).toList();
        return new Detail(summary(row, target), target.projectId(), target.canvasId(), target.capacityHeld(),
                target.remote(), history, kind == Kind.GENERATION ? generationDetails.get(id) : null);
    }

    private Summary summary(ExecutionRecordMapper.Row row, Target target) {
        boolean unknown = target.status().equals("UNKNOWN_SUBMISSION")
                && target.remote().stream().anyMatch(remote -> !remote.responseRecorded());
        return new Summary(row.kind(), row.id(), row.title(), target.status(),
                !target.capacityHeld() && !ACTIVE.contains(target.status()), unknown, safeMessage(row.message()),
                row.createdAt().toInstant(ZoneOffset.UTC), row.updatedAt().toInstant(ZoneOffset.UTC));
    }

    private String safeMessage(String message) {
        return message == null ? null : sanitizer.sanitize(TextNode.valueOf(message)).asText();
    }
}
