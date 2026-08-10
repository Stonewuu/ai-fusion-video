package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.time.LocalDateTime;
import java.util.List;

import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentStateRetentionCleanerTests {

    private final AgentRuntimeSchedulers schedulers = schedulers();
    private final AgentConversationMapper conversations =
            mock(AgentConversationMapper.class);
    private final AgentRunMapper runs = mock(AgentRunMapper.class);
    private final AgentStatePreflight statePreflight = mock(AgentStatePreflight.class);
    private final AgentStateRetentionCleaner cleaner = new AgentStateRetentionCleaner(
            conversations, runs, statePreflight, schedulers);

    @AfterEach
    void closeSchedulers() {
        schedulers.close();
    }

    @Test
    void deletesEveryRootAndChildSessionBeforeMarkingTheConversationExpired() {
        LocalDateTime now = LocalDateTime.of(2026, 7, 30, 12, 0);
        AgentConversation conversation = AgentConversation.builder()
                .id(17L)
                .conversationId("conversation-7")
                .userId(42L)
                .agentStateLastActiveAt(now.minusDays(31))
                .build();
        when(conversations.selectDatabaseNow()).thenReturn(now);
        when(conversations.selectAgentStateCleanupCandidates(
                now.minusDays(30), 100)).thenReturn(List.of(conversation));
        when(runs.selectStateSessionIdsByConversation("conversation-7"))
                .thenReturn(List.of(
                        "afv-child:child-session",
                        "afv:v2:conversation-7:assistant"));
        when(statePreflight.deleteSession("42", "afv-child:child-session"))
                .thenReturn(Mono.empty());
        when(statePreflight.deleteSession(
                "42", "afv:v2:conversation-7:assistant"))
                .thenReturn(Mono.empty());
        when(conversations.markAgentStateExpired(
                17L, conversation.getAgentStateLastActiveAt(), now))
                .thenReturn(1);

        StepVerifier.create(cleaner.cleanExpired(30, 100))
                .expectNext(1L)
                .verifyComplete();

        var ordered = inOrder(statePreflight, conversations);
        ordered.verify(statePreflight).deleteSession(
                "42", "afv-child:child-session");
        ordered.verify(statePreflight).deleteSession(
                "42", "afv:v2:conversation-7:assistant");
        ordered.verify(conversations).markAgentStateExpired(
                17L, conversation.getAgentStateLastActiveAt(), now);
    }

    @Test
    void stateStoreFailureNeverMarksTheConversationExpired() {
        LocalDateTime now = LocalDateTime.of(2026, 7, 30, 12, 0);
        AgentConversation conversation = AgentConversation.builder()
                .id(17L)
                .conversationId("conversation-7")
                .userId(42L)
                .agentStateLastActiveAt(now.minusDays(31))
                .build();
        RuntimeException failure = new RuntimeException("state store unavailable");
        when(conversations.selectDatabaseNow()).thenReturn(now);
        when(conversations.selectAgentStateCleanupCandidates(
                now.minusDays(30), 100)).thenReturn(List.of(conversation));
        when(runs.selectStateSessionIdsByConversation("conversation-7"))
                .thenReturn(List.of("afv:v2:conversation-7:assistant"));
        when(statePreflight.deleteSession(
                "42", "afv:v2:conversation-7:assistant"))
                .thenReturn(Mono.error(failure));

        StepVerifier.create(cleaner.cleanExpired(30, 100))
                .expectErrorMatches(actual -> actual == failure)
                .verify();

        verify(conversations, never()).markAgentStateExpired(
                17L, conversation.getAgentStateLastActiveAt(), now);
    }

    private AgentRuntimeSchedulers schedulers() {
        AgentScopeRuntimeProperties properties = new AgentScopeRuntimeProperties();
        properties.setStateThreads(1);
        properties.setJournalThreads(1);
        properties.setModelThreads(1);
        properties.setToolThreads(1);
        return new AgentRuntimeSchedulers(properties);
    }
}
