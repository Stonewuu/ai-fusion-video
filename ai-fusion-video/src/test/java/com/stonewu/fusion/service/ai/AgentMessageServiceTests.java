package com.stonewu.fusion.service.ai;

import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentMessageServiceTests {

    @Test
    void saveAssistantMessage_shouldPersistReasoningOnlyMessage() {
        AgentMessageMapper mapper = mock(AgentMessageMapper.class);
        when(mapper.findMaxMessageOrder("conv-1")).thenReturn(3);
        AgentMessageService service = new AgentMessageService(mapper);

        AgentMessage saved = service.saveAssistantMessage("conv-1", "", "先分析工具参数", 1200L);

        ArgumentCaptor<AgentMessage> captor = ArgumentCaptor.forClass(AgentMessage.class);
        verify(mapper).insert(captor.capture());
        AgentMessage inserted = captor.getValue();

        assertThat(saved).isNotNull();
        assertThat(inserted.getConversationId()).isEqualTo("conv-1");
        assertThat(inserted.getRole()).isEqualTo("assistant");
        assertThat(inserted.getContent()).isNull();
        assertThat(inserted.getReasoningContent()).isEqualTo("先分析工具参数");
        assertThat(inserted.getReasoningDurationMs()).isEqualTo(1200L);
        assertThat(inserted.getMessageOrder()).isEqualTo(4);
    }

    @Test
    void saveAssistantMessage_shouldSkipCompletelyEmptyMessage() {
        AgentMessageMapper mapper = mock(AgentMessageMapper.class);
        AgentMessageService service = new AgentMessageService(mapper);

        AgentMessage saved = service.saveAssistantMessage("conv-1", "", "", null);

        assertThat(saved).isNull();
        verify(mapper, never()).findMaxMessageOrder("conv-1");
        verify(mapper, never()).insert(any(AgentMessage.class));
    }
}