package com.stonewu.fusion.service.ai.agentscope.context;

import io.agentscope.core.agent.RuntimeContext;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.time.Instant;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AgentScopeRuntimeContextFactoryTests {

    private static final Instant DEADLINE = Instant.parse("2026-07-21T12:00:00Z");

    private final AgentScopeRuntimeContextFactory factory = new AgentScopeRuntimeContextFactory();

    @Test
    void createsCanonicalAddressAndAllTypedContexts() {
        ProjectContext project = new ProjectContext(21L);
        ToolExecutionContext tool = new ToolExecutionContext(42L, 1, 42L);
        AgentScopeRuntimeContextRequest request = request(project, tool);

        RuntimeContext context = factory.create(request);

        assertThat(context.getUserId()).isEqualTo("42");
        assertThat(context.getSessionId()).isEqualTo("afv:v2:conversation-7:assistant-v3");
        assertThat(context.get(AuthenticatedUserContext.class)).isSameAs(request.authenticatedUser());
        assertThat(context.get(AgentConversationContext.class)).isSameAs(request.conversation());
        assertThat(context.get(AgentRunContext.class)).isSameAs(request.run());
        assertThat(context.get(PipelineRequestContext.class)).isSameAs(request.pipelineRequest());
        assertThat(context.get(CancellationContext.class)).isSameAs(request.cancellation());
        assertThat(context.get(ProjectContext.class)).isSameAs(project);
        assertThat(context.get(ToolExecutionContext.class)).isSameAs(tool);
        assertThat(context.get(AgentRunContext.class).ownerEpoch()).isEqualTo(3L);
    }

    @Test
    void omitsOptionalContextsAndCreatesIndependentPerCallContainers() {
        AgentScopeRuntimeContextRequest request = request(null, null);

        RuntimeContext first = factory.create(request);
        RuntimeContext second = factory.create(request);

        assertThat(first).isNotSameAs(second);
        assertThat(first.get(ProjectContext.class)).isNull();
        assertThat(first.get(ToolExecutionContext.class)).isNull();
        assertThat(second.get(ProjectContext.class)).isNull();
        assertThat(second.get(ToolExecutionContext.class)).isNull();

        first.put(ProjectContext.class, new ProjectContext(99L));

        assertThat(first.get(ProjectContext.class).projectId()).isEqualTo(99L);
        assertThat(second.get(ProjectContext.class)).isNull();
    }

    @Test
    void normalizesIdentifiersAndRejectsAmbiguousSessionComponents() {
        AgentScopeRuntimeContextRequest normalized = new AgentScopeRuntimeContextRequest(
                new AuthenticatedUserContext(42L),
                new AgentConversationContext("  conversation-7  ", "  assistant-v3  "),
                new AgentRunContext("  run-9  ", "  node-a  ", 3L, DEADLINE),
                null,
                new PipelineRequestContext("  request-11  ", PipelineRequestContext.Kind.PIPELINE),
                null,
                CancellationContext.noop());

        RuntimeContext context = factory.create(normalized);

        assertThat(context.getSessionId()).isEqualTo("afv:v2:conversation-7:assistant-v3");
        assertThat(context.get(AgentRunContext.class).runId()).isEqualTo("run-9");
        assertThat(context.get(PipelineRequestContext.class).requestId()).isEqualTo("request-11");
        assertThatThrownBy(() -> new AgentConversationContext("conversation:7", "assistant-v3"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("conversationId");
        assertThatThrownBy(() -> new AgentConversationContext("conversation-7", "assistant:v3"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("agentDefinitionStableKey");
    }

    @Test
    void rejectsInvalidRequiredContextBeforeBuildingAgentScopeStateAddress() {
        assertThatThrownBy(() -> factory.create(null))
                .isInstanceOf(NullPointerException.class)
                .hasMessageContaining("request");
        assertThatThrownBy(() -> new AuthenticatedUserContext(0L))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("userId");
        assertThatThrownBy(() -> new AgentConversationContext(" ", "assistant-v3"))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("conversationId");
        assertThatThrownBy(() -> new AgentRunContext("run-9", "node-a", 0L, DEADLINE))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("ownerEpoch");
        assertThatThrownBy(() -> new ProjectContext(-1L))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("projectId");
        assertThatThrownBy(() -> new ToolExecutionContext(42L, 0, 42L))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("ownerType");
        assertThatThrownBy(() -> new PipelineRequestContext(" ", PipelineRequestContext.Kind.CHAT))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("requestId");
        assertThatThrownBy(() -> new AgentScopeRuntimeContextRequest(
                        new AuthenticatedUserContext(42L),
                        new AgentConversationContext("conversation-7", "assistant-v3"),
                        null,
                        null,
                        new PipelineRequestContext("request-11", PipelineRequestContext.Kind.CHAT),
                        null,
                        CancellationContext.noop()))
                .isInstanceOf(NullPointerException.class)
                .hasMessageContaining("run");
    }

    @Test
    void cancellationCheckpointIsLazyAndPreservesReactiveFailure() {
        AtomicInteger subscriptions = new AtomicInteger();
        IllegalStateException failure = new IllegalStateException("cancelled");
        CancellationContext cancellation = new CancellationContext(
                () -> Mono.defer(() -> {
                    subscriptions.incrementAndGet();
                    return Mono.error(failure);
                }));

        assertThat(subscriptions).hasValue(0);
        StepVerifier.create(cancellation.checkpoint())
                .expectErrorSatisfies(actual -> assertThat(actual).isSameAs(failure))
                .verify();
        assertThat(subscriptions).hasValue(1);
    }

    private AgentScopeRuntimeContextRequest request(
            ProjectContext project, ToolExecutionContext toolExecution) {
        return new AgentScopeRuntimeContextRequest(
                new AuthenticatedUserContext(42L),
                new AgentConversationContext("conversation-7", "assistant-v3"),
                new AgentRunContext("run-9", "node-a", 3L, DEADLINE),
                project,
                new PipelineRequestContext("request-11", PipelineRequestContext.Kind.PIPELINE),
                toolExecution,
                CancellationContext.noop());
    }
}
