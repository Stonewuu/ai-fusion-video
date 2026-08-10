package com.stonewu.fusion.service.ai.run;

import com.stonewu.fusion.service.ai.run.model.NormalizedModelUsage;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class AuditLedgerModelUsageSettlementAdapterTests {

    private final AuditLedgerModelUsageSettlementAdapter adapter =
            new AuditLedgerModelUsageSettlementAdapter();

    @Test
    void returnsStableDistinctSettlementIdsForCallKeys() {
        NormalizedModelUsage usage = NormalizedModelUsage.tokens(12, 7);

        String first = adapter.settle("run-1:call-1", usage).block();
        String repeated = adapter.settle("run-1:call-1", usage).block();
        String second = adapter.settle("run-1:call-2", usage).block();

        assertThat(first)
                .matches("usage-[0-9a-f]{64}")
                .isEqualTo(repeated)
                .isNotEqualTo(second);
    }

    @Test
    void validatesInputsAtSubscriptionTime() {
        Mono<String> invalidKey = adapter.settle(
                "missing-separator", NormalizedModelUsage.tokens(1, 1));
        Mono<String> missingUsage = adapter.settle("run-1:call-1", null);

        StepVerifier.create(invalidKey)
                .expectErrorSatisfies(error -> assertThat(error)
                        .isInstanceOf(IllegalArgumentException.class)
                        .hasMessage("idempotency key must be runId:modelCallId"))
                .verify();
        StepVerifier.create(missingUsage)
                .expectErrorSatisfies(error -> assertThat(error)
                        .isInstanceOf(NullPointerException.class)
                        .hasMessage("usage must not be null"))
                .verify();
    }
}
