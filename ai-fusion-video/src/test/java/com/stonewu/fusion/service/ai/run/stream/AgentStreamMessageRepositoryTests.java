package com.stonewu.fusion.service.ai.run.stream;

import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class AgentStreamMessageRepositoryTests {
    @Test
    void expiredOwnerCannotSaveCompleteMessagesEvenBeforeAnotherOwnerTakesOver() {
        AgentRunMapper runs = mock(AgentRunMapper.class);
        AgentConversationMapper conversations = mock(AgentConversationMapper.class);
        AgentMessageMapper messages = mock(AgentMessageMapper.class);
        LocalDateTime now = LocalDateTime.of(2026, 11, 24, 9, 0);
        when(runs.selectDatabaseNow()).thenReturn(now);
        when(runs.selectByRunIdForUpdate("run")).thenReturn(AgentRun.builder().runId("run")
                .ownerInstanceId("owner").ownerEpoch(1L).status("RUNNING")
                .deadlineAt(now.plusHours(1)).leaseUntil(now).build());
        AgentStreamMessageRepository repository = new AgentStreamMessageRepository(runs, conversations, messages);

        assertThatThrownBy(() -> repository.save("run", "owner", 1,
                List.of(AgentMessage.builder().role("assistant").content("完整回复").build())))
                .isInstanceOf(IllegalStateException.class).hasMessageContaining("不能保存");
        verifyNoInteractions(conversations, messages);
    }
}
