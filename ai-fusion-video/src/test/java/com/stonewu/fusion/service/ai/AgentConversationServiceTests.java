package com.stonewu.fusion.service.ai;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.entity.ai.AgentStateCleanupPolicy;
import com.stonewu.fusion.mapper.ai.AgentConversationMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStatePreflight;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStateCleanupPolicyService;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.InOrder;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.test.StepVerifier;

import java.util.List;
import java.time.LocalDateTime;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class AgentConversationServiceTests {

    @BeforeAll
    static void initMybatisPlusTableInfo() {
        TableInfoHelper.initTableInfo(
                new MapperBuilderAssistant(new MybatisConfiguration(), ""), AgentConversation.class);
    }

    private final AgentRuntimeSchedulers schedulers = schedulers();
    private final AgentConversationMapper conversationMapper = mock(AgentConversationMapper.class);
    private final AgentStatePreflight statePreflight = mock(AgentStatePreflight.class);
    private final AgentStateCleanupPolicyService stateCleanupPolicy =
            mock(AgentStateCleanupPolicyService.class);
    private final AgentRunMapper runMapper = mock(AgentRunMapper.class);
    private final AgentConversationService conversationService =
            new AgentConversationService(
                    conversationMapper,
                    statePreflight,
                    stateCleanupPolicy,
                    runMapper,
                    schedulers);

    @AfterEach
    void closeSchedulers() {
        schedulers.close();
    }

    @Test
    void ownedConversationCleansEveryRuntimeSessionBeforeDeletingTheOwnedRow() {
        AgentConversation conversation = AgentConversation.builder()
                .id(17L)
                .userId(42L)
                .conversationId("conversation-7")
                .build();
        when(conversationMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(conversation);
        when(runMapper.selectStateSessionIdsByConversation("conversation-7"))
                .thenReturn(List.of(
                        "afv-child:child-session",
                        "afv:v2:conversation-7:assistant"));
        when(statePreflight.deleteSession("42", "afv-child:child-session"))
                .thenReturn(Mono.empty());
        when(statePreflight.deleteSession(
                "42", "afv:v2:conversation-7:assistant"))
                .thenReturn(Mono.empty());
        List<String> phases = new java.util.concurrent.CopyOnWriteArrayList<>();
        doAnswer(ignored -> {
            phases.add("delete-owned-row");
            return 1;
        }).when(conversationMapper).delete(any(LambdaQueryWrapper.class));

        StepVerifier.create(conversationService.deleteConversation(17L, 42L))
                .verifyComplete();

        ArgumentCaptor<LambdaQueryWrapper<AgentConversation>> selectCaptor = lambdaWrapperCaptor();
        ArgumentCaptor<LambdaQueryWrapper<AgentConversation>> deleteCaptor = lambdaWrapperCaptor();
        InOrder ordered = inOrder(conversationMapper, runMapper, statePreflight);
        ordered.verify(conversationMapper).selectOne(selectCaptor.capture());
        ordered.verify(runMapper).selectStateSessionIdsByConversation("conversation-7");
        ordered.verify(statePreflight).deleteSession("42", "afv-child:child-session");
        ordered.verify(statePreflight).deleteSession(
                "42", "afv:v2:conversation-7:assistant");
        ordered.verify(conversationMapper).delete(deleteCaptor.capture());
        assertOwnedRowPredicate(selectCaptor.getValue(), 17L, 42L);
        assertOwnedRowPredicate(deleteCaptor.getValue(), 17L, 42L);
        assertThat(phases).containsExactly("delete-owned-row");
    }

    @Test
    void cleanupFailureLeavesTheConversationRowIntactForRetry() {
        AgentConversation conversation = AgentConversation.builder()
                .id(17L)
                .userId(42L)
                .conversationId("conversation-7")
                .build();
        RuntimeException cleanupFailure = new RuntimeException("state store unavailable");
        when(conversationMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(conversation);
        when(runMapper.selectStateSessionIdsByConversation("conversation-7"))
                .thenReturn(List.of("afv:v2:conversation-7:assistant"));
        when(statePreflight.deleteSession(
                "42", "afv:v2:conversation-7:assistant"))
                .thenReturn(Mono.error(cleanupFailure));

        StepVerifier.create(conversationService.deleteConversation(17L, 42L))
                .expectErrorSatisfies(actual -> assertThat(actual).isSameAs(cleanupFailure))
                .verify();

        verify(conversationMapper, never()).delete(any(LambdaQueryWrapper.class));
    }

    @Test
    void incompleteCleanupCannotDeleteTheConversationRow() {
        AgentConversation conversation = AgentConversation.builder()
                .id(17L)
                .userId(42L)
                .conversationId("conversation-7")
                .build();
        Sinks.Empty<Void> cleanup = Sinks.empty();
        CountDownLatch cleanupSubscribed = new CountDownLatch(1);
        when(conversationMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(conversation);
        when(runMapper.selectStateSessionIdsByConversation("conversation-7"))
                .thenReturn(List.of("afv:v2:conversation-7:assistant"));
        when(statePreflight.deleteSession(
                "42", "afv:v2:conversation-7:assistant"))
                .thenReturn(cleanup.asMono().doOnSubscribe(ignored -> cleanupSubscribed.countDown()));

        StepVerifier.create(conversationService.deleteConversation(17L, 42L))
                .expectSubscription()
                .then(() -> await(cleanupSubscribed))
                .then(() -> verify(conversationMapper, never()).delete(any(LambdaQueryWrapper.class)))
                .thenCancel()
                .verify();
    }

    @Test
    void missingOrOtherOwnerConversationCompletesIdempotentlyWithoutLeakingItsExistence() {
        when(conversationMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(null);

        StepVerifier.create(conversationService.deleteConversation(17L, 42L))
                .verifyComplete();

        ArgumentCaptor<LambdaQueryWrapper<AgentConversation>> selectCaptor = lambdaWrapperCaptor();
        verify(conversationMapper).selectOne(selectCaptor.capture());
        assertOwnedRowPredicate(selectCaptor.getValue(), 17L, 42L);
        verifyNoInteractions(statePreflight);
        verify(conversationMapper, never()).delete(any(LambdaQueryWrapper.class));
    }

    @Test
    void deletesOwnedConversationByStableConversationId() {
        AgentConversation conversation = AgentConversation.builder()
                .id(17L)
                .userId(42L)
                .conversationId("optimistic-conversation")
                .build();
        when(conversationMapper.selectOne(any(LambdaQueryWrapper.class))).thenReturn(conversation);
        when(runMapper.selectStateSessionIdsByConversation("optimistic-conversation"))
                .thenReturn(List.of());

        StepVerifier.create(conversationService.deleteConversationByConversationId(
                        "optimistic-conversation", 42L))
                .verifyComplete();

        ArgumentCaptor<LambdaQueryWrapper<AgentConversation>> selectCaptor = lambdaWrapperCaptor();
        ArgumentCaptor<LambdaQueryWrapper<AgentConversation>> deleteCaptor = lambdaWrapperCaptor();
        InOrder ordered = inOrder(conversationMapper, runMapper, statePreflight);
        ordered.verify(conversationMapper).selectOne(selectCaptor.capture());
        ordered.verify(runMapper).selectStateSessionIdsByConversation(
                "optimistic-conversation");
        ordered.verify(conversationMapper).delete(deleteCaptor.capture());
        assertOwnedConversationPredicate(
                selectCaptor.getValue(), "optimistic-conversation", 42L);
        assertOwnedConversationPredicate(
                deleteCaptor.getValue(), "optimistic-conversation", 42L);
    }

    @Test
    void laterAssistantTurnsDoNotOverwriteTheFirstPromptTitle() {
        AgentConversation conversation = AgentConversation.builder()
                .id(17L)
                .userId(42L)
                .conversationId("conversation-7")
                .category("assistant")
                .title("第一条用户提示")
                .status("completed")
                .build();
        LocalDateTime now = LocalDateTime.of(2026, 7, 30, 12, 0);
        when(conversationMapper.selectByConversationIdForUpdate("conversation-7"))
                .thenReturn(conversation);
        when(conversationMapper.selectDatabaseNow()).thenReturn(now);
        when(stateCleanupPolicy.getCurrent()).thenReturn(
                AgentStateCleanupPolicy.builder().retentionDays(30).build());
        when(conversationMapper.update(isNull(), any(LambdaUpdateWrapper.class))).thenReturn(1);

        AgentConversation updated = conversationService.createOrUpdate(
                "conversation-7",
                42L,
                null,
                null,
                null,
                "ai_assistant_agent",
                "第二条用户提示",
                "assistant");

        assertThat(updated.getTitle()).isEqualTo("第一条用户提示");
        ArgumentCaptor<LambdaUpdateWrapper<AgentConversation>> updateCaptor = updateWrapperCaptor();
        verify(conversationMapper).update(isNull(), updateCaptor.capture());
        assertThat(updateCaptor.getValue().getSqlSet())
                .doesNotContain("title")
                .contains("status");
    }

    private static void assertOwnedRowPredicate(
            LambdaQueryWrapper<AgentConversation> wrapper, long id, long userId) {
        assertThat(wrapper.getSqlSegment())
                .containsPattern("(?i)(^|\\W)id\\s*=")
                .containsPattern("(?i)(^|\\W)user_id\\s*=");
        assertThat(wrapper.getParamNameValuePairs().values()).contains(id, userId);
    }

    private static void assertOwnedConversationPredicate(
            LambdaQueryWrapper<AgentConversation> wrapper,
            String conversationId,
            long userId) {
        assertThat(wrapper.getSqlSegment())
                .containsPattern("(?i)(^|\\W)conversation_id\\s*=")
                .containsPattern("(?i)(^|\\W)user_id\\s*=");
        assertThat(wrapper.getParamNameValuePairs().values())
                .contains(conversationId, userId);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static ArgumentCaptor<LambdaQueryWrapper<AgentConversation>> lambdaWrapperCaptor() {
        return ArgumentCaptor.forClass((Class) LambdaQueryWrapper.class);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private static ArgumentCaptor<LambdaUpdateWrapper<AgentConversation>> updateWrapperCaptor() {
        return ArgumentCaptor.forClass((Class) LambdaUpdateWrapper.class);
    }

    private static void await(CountDownLatch latch) {
        try {
            assertThat(latch.await(5, TimeUnit.SECONDS)).isTrue();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError("interrupted while waiting for cleanup subscription", interrupted);
        }
    }

    private static AgentRuntimeSchedulers schedulers() {
        AgentScopeRuntimeProperties properties = new AgentScopeRuntimeProperties();
        properties.setStateThreads(1);
        properties.setJournalThreads(1);
        properties.setModelThreads(1);
        properties.setToolThreads(1);
        return new AgentRuntimeSchedulers(properties);
    }
}
