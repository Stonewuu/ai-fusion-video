package com.stonewu.fusion.repository.ai;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.stonewu.fusion.enums.ai.AgentRunStatus;

/**
 * Synchronous persistence boundary used only inside short journal transactions.
 */
@Repository
@RequiredArgsConstructor
public class AgentRunRepository {

    private final AgentRunMapper runMapper;
    private final AgentConversationMapper conversationMapper;
    private final AgentMessageMapper messageMapper;

    public AgentConversation lockConversation(String conversationId) {
        return conversationMapper.selectByConversationIdForUpdate(conversationId);
    }

    public void activateConversation(
            AgentConversation conversation,
            LocalDateTime databaseNow) {
        conversation.setStatus("running");
        conversation.setAgentStateLastActiveAt(databaseNow);
        conversation.setAgentStateExpiredAt(null);
        conversation.setUpdateTime(databaseNow);
        if (conversationMapper.updateById(conversation) != 1) {
            throw new IllegalStateException(
                    "Agent conversation activation did not affect exactly one row");
        }
    }

    public AgentRun lockRun(String runId) {
        return runMapper.selectByRunIdForUpdate(runId);
    }

    public AgentRun findRun(String runId) {
        return runMapper.selectByRunId(runId);
    }

    public AgentRun findLatestRoot(String conversationId) {
        return runMapper.selectLatestRootByConversation(conversationId);
    }

    public AgentRun lockChild(String parentRunId, String parentToolCallId) {
        return runMapper.selectByParentAndToolCallForUpdate(parentRunId, parentToolCallId);
    }

    public List<AgentRun> findActiveChildren(String parentRunId) {
        return runMapper.selectActiveChildren(parentRunId);
    }

    public LocalDateTime databaseNow() {
        return runMapper.selectDatabaseNow();
    }

    public Long findInitialMessageOrder(String runId) {
        return messageMapper.selectInitialOrderByRunId(runId);
    }

    public void insert(AgentRun run) {
        if (runMapper.insert(run) != 1) {
            throw new IllegalStateException("Agent run insert did not affect exactly one row");
        }
    }

    public void update(AgentRun run) {
        if (runMapper.updateById(run) != 1) {
            throw new IllegalStateException("Agent run update did not affect exactly one row");
        }
    }

    public AgentRun requestCancellation(String runId) {
        return requestInternalCancellationTree(runId).root();
    }

    public CancellationTree requestAuthorizedCancellationTree(
            String runId, long currentUserId) {
        AgentRun root = lockRun(runId);
        if (root == null) {
            throw new BusinessException(404, "Agent 运行不存在");
        }
        if (!java.util.Objects.equals(root.getUserId(), currentUserId)) {
            throw new BusinessException(403, "无权取消该 Agent 运行");
        }
        return requestCancellationTree(root, true);
    }

    public CancellationTree requestInternalCancellationTree(String runId) {
        AgentRun root = lockRun(runId);
        if (root == null) {
            throw new BusinessException(404, "Agent 运行不存在");
        }
        return requestCancellationTree(root, true);
    }

    public CancellationTree requestDescendantCancellationTree(String parentRunId) {
        AgentRun parent = lockRun(parentRunId);
        if (parent == null) {
            throw new IllegalStateException(
                    "Parent Agent run does not exist: " + parentRunId);
        }
        List<AgentRun> affected = new ArrayList<>();
        requestCancelActiveDescendants(
                parent, databaseNow(), affected, new HashSet<>());
        return new CancellationTree(parent, affected);
    }

    public List<AgentRun> requestCancelActiveDescendants(String parentRunId) {
        return requestDescendantCancellationTree(parentRunId).affected();
    }

    public boolean hasValidOwnedLease(
            String runId, String ownerInstanceId, long ownerEpoch) {
        return runMapper.countValidOwnedLease(
                runId, ownerInstanceId, ownerEpoch) == 1;
    }

    public boolean renewOwnedLease(
            String runId,
            String ownerInstanceId,
            long ownerEpoch,
            Duration lease) {
        return runMapper.renewOwnedLease(
                runId,
                ownerInstanceId,
                ownerEpoch,
                leaseMicros(lease)) == 1;
    }

    public boolean acknowledgeOwnedCancellation(
            String runId, String ownerInstanceId, long ownerEpoch) {
        return runMapper.acknowledgeOwnedCancellation(
                runId, ownerInstanceId, ownerEpoch) == 1;
    }

    public boolean markCancellationBroadcast(
            String runId, Duration retryDelay) {
        LocalDateTime now = databaseNow();
        return runMapper.markCancellationBroadcast(
                runId, now, now.plus(retryDelay)) == 1;
    }

    public List<AgentRun> findExpiredLeaseCandidates(int limit) {
        requireBatchLimit(limit);
        return List.copyOf(runMapper.selectExpiredLeaseCandidates(
                databaseNow(), limit));
    }

    public List<AgentRun> findCancellationRetryCandidates(int limit) {
        requireBatchLimit(limit);
        return List.copyOf(runMapper.selectCancellationRetryCandidates(
                databaseNow(), limit));
    }

    private CancellationTree requestCancellationTree(
            AgentRun root, boolean includeRoot) {
        LocalDateTime now = databaseNow();
        List<AgentRun> affected = new ArrayList<>();
        if (includeRoot && requestCancellation(root, now)) {
            affected.add(root);
        }
        if (!AgentRunStatus.valueOf(root.getStatus()).isTerminal()) {
            requestCancelActiveDescendants(root, now, affected, new HashSet<>());
        }
        return new CancellationTree(root, affected);
    }

    private boolean requestCancellation(AgentRun run, LocalDateTime now) {
        AgentRunStatus status = AgentRunStatus.valueOf(run.getStatus());
        if (status.isTerminal()) {
            return false;
        }
        if (status != AgentRunStatus.CANCEL_REQUESTED) {
            run.setStatus(AgentRunStatus.CANCEL_REQUESTED.name());
            run.setCancelRequestedAt(now);
            run.setCancelNextAttemptAt(now);
            update(run);
        } else if (run.getCancelNextAttemptAt() == null) {
            run.setCancelNextAttemptAt(now);
            update(run);
        }
        return true;
    }

    private void requestCancelActiveDescendants(
            AgentRun parent,
            LocalDateTime now,
            List<AgentRun> affected,
            Set<String> visited) {
        if (!visited.add(parent.getRunId())) {
            throw new IllegalStateException("Cycle detected in Agent child runs");
        }
        for (AgentRun candidate : findActiveChildren(parent.getRunId())) {
            AgentRun child = lockRun(candidate.getRunId());
            if (child == null || AgentRunStatus.valueOf(child.getStatus()).isTerminal()) {
                continue;
            }
            if (requestCancellation(child, now)) {
                affected.add(child);
            }
            requestCancelActiveDescendants(child, now, affected, visited);
        }
    }

    private long leaseMicros(Duration lease) {
        if (lease == null || lease.isZero() || lease.isNegative()) {
            throw new IllegalArgumentException("lease must be greater than zero");
        }
        try {
            return Math.max(1L, lease.toNanos() / 1_000L);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("lease is too large", overflow);
        }
    }

    private void requireBatchLimit(int limit) {
        if (limit <= 0 || limit > 1000) {
            throw new IllegalArgumentException("limit must be between 1 and 1000");
        }
    }

    public record CancellationTree(AgentRun root, List<AgentRun> affected) {

        public CancellationTree {
            if (root == null) {
                throw new IllegalArgumentException("root must not be null");
            }
            affected = List.copyOf(affected);
        }
    }
}
