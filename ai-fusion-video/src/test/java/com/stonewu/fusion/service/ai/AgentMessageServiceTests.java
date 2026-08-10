package com.stonewu.fusion.service.ai;

import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.service.ai.run.AgentMessageAllocator;
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
        AgentMessageAllocator allocator = mock(AgentMessageAllocator.class);
        when(allocator.append(org.mockito.ArgumentMatchers.eq("conv-1"),
                any(AgentMessage.class))).thenAnswer(invocation -> {
                    AgentMessage message = invocation.getArgument(1);
                    message.setConversationId("conv-1");
                    message.setMessageOrder(4L);
                    return 4L;
                });
        AgentMessageService service = new AgentMessageService(mapper, allocator);

        AgentMessage saved = service.saveAssistantMessage("conv-1", "", "先分析工具参数", 1200L);

        ArgumentCaptor<AgentMessage> captor = ArgumentCaptor.forClass(AgentMessage.class);
        verify(allocator).append(org.mockito.ArgumentMatchers.eq("conv-1"), captor.capture());
        AgentMessage inserted = captor.getValue();

        assertThat(saved).isNotNull();
        assertThat(inserted.getConversationId()).isEqualTo("conv-1");
        assertThat(inserted.getRole()).isEqualTo("assistant");
        assertThat(inserted.getContent()).isNull();
        assertThat(inserted.getReasoningContent()).isEqualTo("先分析工具参数");
        assertThat(inserted.getReasoningDurationMs()).isEqualTo(1200L);
        assertThat(inserted.getMessageOrder()).isEqualTo(4L);
    }

    @Test
    void saveAssistantMessage_shouldSkipCompletelyEmptyMessage() {
        AgentMessageMapper mapper = mock(AgentMessageMapper.class);
        AgentMessageAllocator allocator = mock(AgentMessageAllocator.class);
        AgentMessageService service = new AgentMessageService(mapper, allocator);

        AgentMessage saved = service.saveAssistantMessage("conv-1", "", "", null);

        assertThat(saved).isNull();
        verify(mapper, never()).insert(any(AgentMessage.class));
        verify(allocator, never()).append(org.mockito.ArgumentMatchers.anyString(),
                any(AgentMessage.class));
    }
}
