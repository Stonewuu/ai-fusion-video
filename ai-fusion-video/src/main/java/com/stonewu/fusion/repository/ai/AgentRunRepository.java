package com.stonewu.fusion.repository.ai;

import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;
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

    public AgentRun lockRun(String runId) {
        return runMapper.selectByRunIdForUpdate(runId);
    }

    public AgentRun findRun(String runId) {
        return runMapper.selectByRunId(runId);
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
        AgentRun run = lockRun(runId);
        if (run == null || AgentRunStatus.valueOf(run.getStatus()).isTerminal()) {
            return run;
        }
        if (!AgentRunStatus.CANCEL_REQUESTED.name().equals(run.getStatus())) {
            LocalDateTime now = databaseNow();
            run.setStatus(AgentRunStatus.CANCEL_REQUESTED.name());
            run.setCancelRequestedAt(now);
            run.setCancelNextAttemptAt(now);
            update(run);
        }
        requestCancelActiveDescendants(run, new ArrayList<>(), new HashSet<>());
        return run;
    }

    public List<AgentRun> requestCancelActiveDescendants(String parentRunId) {
        AgentRun parent = lockRun(parentRunId);
        if (parent == null) {
            throw new IllegalStateException("Parent Agent run does not exist: " + parentRunId);
        }
        List<AgentRun> cancelled = new ArrayList<>();
        requestCancelActiveDescendants(parent, cancelled, new HashSet<>());
        return List.copyOf(cancelled);
    }

    private void requestCancelActiveDescendants(
            AgentRun parent,
            List<AgentRun> cancelled,
            Set<String> visited) {
        if (!visited.add(parent.getRunId())) {
            throw new IllegalStateException("Cycle detected in Agent child runs");
        }
        for (AgentRun candidate : findActiveChildren(parent.getRunId())) {
            AgentRun child = lockRun(candidate.getRunId());
            if (child == null || AgentRunStatus.valueOf(child.getStatus()).isTerminal()) {
                continue;
            }
            if (!AgentRunStatus.CANCEL_REQUESTED.name().equals(child.getStatus())) {
                LocalDateTime now = databaseNow();
                child.setStatus(AgentRunStatus.CANCEL_REQUESTED.name());
                child.setCancelRequestedAt(now);
                child.setCancelNextAttemptAt(now);
                update(child);
            }
            cancelled.add(child);
            requestCancelActiveDescendants(child, cancelled, visited);
        }
    }
}
