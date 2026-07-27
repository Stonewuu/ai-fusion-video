package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.entity.ai.AgentConversation;
import com.stonewu.fusion.security.SecurityUserDetails;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AgentMessageService;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentScopeMcpRegistry;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentScopeSkillRegistry;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentUserSkillService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.test.StepVerifier;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class AiAssistantControllerTests {

    private final AgentConversationService conversationService = mock(AgentConversationService.class);
    private final AgentMessageService messageService = mock(AgentMessageService.class);
    private final AgentScopeSkillRegistry skillRegistry = mock(AgentScopeSkillRegistry.class);
    private final AgentScopeMcpRegistry mcpRegistry = mock(AgentScopeMcpRegistry.class);
    private final AgentUserSkillService userSkillService = mock(AgentUserSkillService.class);
    private final AiAssistantController controller =
            new AiAssistantController(
                    conversationService, messageService, skillRegistry, mcpRegistry, userSkillService);

    @AfterEach
    void clearSecurityContext() {
        SecurityContextHolder.clearContext();
    }

    @Test
    void deleteUsesCurrentSecurityUserAndRespondsOnlyAfterServiceCompletion() {
        SecurityUserDetails user = new SecurityUserDetails(42L, "owner", "secret", 1, null, List.of());
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities()));
        Sinks.Empty<Void> serviceCompletion = Sinks.empty();
        when(conversationService.deleteConversation(17L, 42L)).thenReturn(serviceCompletion.asMono());
        AtomicBoolean responseEmitted = new AtomicBoolean();

        Mono<CommonResult<Boolean>> response = controller.deleteConversation(17L)
                .doOnNext(ignored -> responseEmitted.set(true));

        StepVerifier.create(response)
                .expectSubscription()
                .then(() -> {
                    verify(conversationService).deleteConversation(17L, 42L);
                    assertThat(serviceCompletion.currentSubscriberCount()).isEqualTo(1);
                    assertThat(responseEmitted).isFalse();
                })
                .then(() -> assertThat(serviceCompletion.tryEmitEmpty()).isEqualTo(Sinks.EmitResult.OK))
                .assertNext(result -> {
                    assertThat(result.getCode()).isZero();
                    assertThat(result.getData()).isTrue();
                })
                .verifyComplete();
    }

    @Test
    void deleteByConversationIdUsesCurrentSecurityUser() {
        SecurityUserDetails user = new SecurityUserDetails(42L, "owner", "secret", 1, null, List.of());
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities()));
        when(conversationService.deleteConversationByConversationId(
                "optimistic-conversation", 42L)).thenReturn(Mono.empty());

        StepVerifier.create(controller.deleteConversationByConversationId(
                        "optimistic-conversation"))
                .assertNext(result -> {
                    assertThat(result.getCode()).isZero();
                    assertThat(result.getData()).isTrue();
                })
                .verifyComplete();

        verify(conversationService).deleteConversationByConversationId(
                "optimistic-conversation", 42L);
    }

    @Test
    void referenceOptionsReturnConfiguredSkillAndMcpCatalogs() {
        SecurityUserDetails user = new SecurityUserDetails(42L, "owner", "secret", 1, null, List.of());
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities()));
        when(skillRegistry.catalog()).thenReturn(List.of(
                new AgentScopeSkillRegistry.SkillReference(
                        "fusion-video-workflow_bundled",
                        "fusion-video-workflow",
                        "融光视频工作流",
                        "bundled")));
        when(mcpRegistry.catalogForAgent("ai_assistant_agent")).thenReturn(List.of(
                new AgentScopeMcpRegistry.McpToolReference(
                        "assets", "search_assets", "搜索素材", true)));

        CommonResult<?> result = controller.referenceOptions();

        assertThat(result.getCode()).isZero();
        assertThat(result.getData()).isNotNull();
    }

    @Test
    void listMessagesRejectsAnUnknownOrOtherUsersConversationBeforeReadingMessages() {
        SecurityUserDetails user = new SecurityUserDetails(42L, "owner", "secret", 1, null, List.of());
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities()));
        when(conversationService.getOwnedByConversationId("conversation-other", 42L))
                .thenReturn(null);

        assertThatThrownBy(() -> controller.listMessages("conversation-other"))
                .isInstanceOfSatisfying(BusinessException.class, error -> {
                    assertThat(error.getCode()).isEqualTo(404);
                    assertThat(error.getMessage()).isEqualTo("对话不存在");
                });

        verify(conversationService).getOwnedByConversationId("conversation-other", 42L);
        verifyNoInteractions(messageService);
    }

    @Test
    void listMessagesReadsOnlyAfterOwnershipHasBeenConfirmed() {
        SecurityUserDetails user = new SecurityUserDetails(42L, "owner", "secret", 1, null, List.of());
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken(user, null, user.getAuthorities()));
        AgentConversation conversation = AgentConversation.builder()
                .conversationId("conversation-owned")
                .userId(42L)
                .build();
        when(conversationService.getOwnedByConversationId("conversation-owned", 42L))
                .thenReturn(conversation);
        when(messageService.listByConversation("conversation-owned")).thenReturn(List.of());

        CommonResult<?> result = controller.listMessages("conversation-owned");

        assertThat(result.getCode()).isZero();
        verify(messageService).listByConversation("conversation-owned");
    }
}
