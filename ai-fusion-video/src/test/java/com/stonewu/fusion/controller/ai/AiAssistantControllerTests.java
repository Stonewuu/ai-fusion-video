package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.security.SecurityUserDetails;
import com.stonewu.fusion.service.ai.AgentConversationService;
import com.stonewu.fusion.service.ai.AgentMessageService;
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
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AiAssistantControllerTests {

    private final AgentConversationService conversationService = mock(AgentConversationService.class);
    private final AgentMessageService messageService = mock(AgentMessageService.class);
    private final AiAssistantController controller =
            new AiAssistantController(conversationService, messageService);

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
}
