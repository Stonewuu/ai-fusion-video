package com.stonewu.fusion.service.generation.job;

import org.junit.jupiter.api.Test;
import java.util.concurrent.atomic.AtomicInteger;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class PreparedGenerationCallTests {
    @Test void failedSubmissionCannotSendAgainAndCleanupRunsOnce() {
        var submissions = new AtomicInteger();
        var cleanups = new AtomicInteger();
        var call = new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
            submissions.incrementAndGet();
            throw new IllegalStateException("响应丢失");
        }, cleanups::incrementAndGet);
        assertThat(submissions.get()).isZero();
        try (call) {
            assertThatThrownBy(call::submit).hasMessage("响应丢失");
            assertThatThrownBy(call::submit).hasMessageContaining("不能再次调用");
        }
        call.close();
        assertThat(submissions.get()).isEqualTo(1);
        assertThat(cleanups.get()).isEqualTo(1);
    }

    @Test void cancellationAfterPreparationClosesSdkResourcesWithoutSending() {
        var submissions = new AtomicInteger();
        var cleanups = new AtomicInteger();
        var call = new PreparedGenerationCall(GenerationPreparedRequest.ordinary(), () -> {
            submissions.incrementAndGet();
            return GenerationProviderReceipt.accepted("original");
        }, cleanups::incrementAndGet);
        call.close();
        assertThatThrownBy(call::submit).hasMessageContaining("不能再次调用");
        assertThat(submissions.get()).isZero();
        assertThat(cleanups.get()).isEqualTo(1);
    }
}
