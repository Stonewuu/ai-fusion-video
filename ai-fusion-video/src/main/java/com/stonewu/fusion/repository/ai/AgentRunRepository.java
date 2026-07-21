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
}
