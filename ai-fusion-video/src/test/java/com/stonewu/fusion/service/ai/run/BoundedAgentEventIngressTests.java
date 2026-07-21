package com.stonewu.fusion.service.ai.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.service.ai.run.model.AgentEventEnvelope;
import org.junit.jupiter.api.Test;
import reactor.test.StepVerifier;

import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;

class BoundedAgentEventIngressTests {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void rejectsCountOverflowWithoutGrowingPastTheConfiguredBound() {
        BoundedAgentEventIngress ingress = new BoundedAgentEventIngress(
                objectMapper, 2, 1024 * 1024);
        AgentEventEnvelope first = event("1", "a");
        AgentEventEnvelope second = event("2", "b");

        assertThat(ingress.offer(first)).isTrue();
        assertThat(ingress.offer(second)).isTrue();
        assertThat(ingress.offer(event("3", "c"))).isFalse();
        assertThat(ingress.queuedEvents()).isEqualTo(2);

        ingress.complete();
        StepVerifier.create(ingress.events())
                .expectNext(first, second)
                .verifyComplete();
        assertThat(ingress.queuedEvents()).isZero();
        assertThat(ingress.queuedBytes()).isZero();
    }

    @Test
    void rejectsByteOverflowAndNeverAdmitsAfterCompletion() {
        BoundedAgentEventIngress ingress = new BoundedAgentEventIngress(
                objectMapper, 10, 1);

        assertThat(ingress.offer(event("1", "payload"))).isFalse();
        assertThat(ingress.queuedEvents()).isZero();
        ingress.complete();
        assertThat(ingress.offer(event("2", "late"))).isFalse();
        StepVerifier.create(ingress.events()).verifyComplete();
    }

    @Test
    void permitsExactlyOneDurableConsumer() {
        BoundedAgentEventIngress ingress = new BoundedAgentEventIngress(
                objectMapper, 10, 1024 * 1024);
        ingress.complete();

        StepVerifier.create(ingress.events()).verifyComplete();
        StepVerifier.create(ingress.events())
                .expectErrorMessage(
                        "Agent event ingress supports exactly one durable consumer")
                .verify();
    }

    private AgentEventEnvelope event(String id, String delta) {
        return new AgentEventEnvelope(
                id,
                "TEXT_BLOCK_DELTA",
                "main",
                "reply",
                "block",
                null,
                null,
                null,
                "CONTENT",
                JsonNodeFactory.instance.objectNode().put("delta", delta),
                Instant.parse("2026-07-21T00:00:00Z"));
    }
}
