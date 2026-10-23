package com.stonewu.fusion.controller.ai;

import com.stonewu.fusion.config.AgentPipelineSseFilter;
import org.junit.jupiter.api.Test;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

class AgentPipelineStreamSupportTests {
    @Test
    void heartbeatsDoNotSubscribeToTheRunTwiceAndStopWithItsTerminalEvent() {
        AtomicInteger subscriptions = new AtomicInteger();
        StepVerifier.withVirtualTime(() -> AgentPipelineStreamSupport.keepAlive(
                        Mono.delay(Duration.ofSeconds(31)).map(tick -> ServerSentEvent.builder("DONE").id("run:1").build())
                                .flux().doOnSubscribe(ignored -> subscriptions.incrementAndGet())))
                .thenAwait(Duration.ofSeconds(15)).assertNext(event -> {
                    assertThat(event.comment()).isEqualTo("keep-alive");
                    assertThat(event.id()).isNull();
                })
                .thenAwait(Duration.ofSeconds(15)).expectNextCount(1)
                .thenAwait(Duration.ofSeconds(1)).assertNext(event -> assertThat(event.data()).isEqualTo("DONE"))
                .verifyComplete();
        assertThat(subscriptions).hasValue(1);
    }

    @Test
    void sourceFailuresStillCloseTheHeartbeat() {
        StepVerifier.create(AgentPipelineStreamSupport.keepAlive(Flux.error(new IllegalStateException("失去租约"))))
                .expectErrorMessage("失去租约").verify(Duration.ofSeconds(1));
    }

    @Test
    void pipelineConnectionsDisableBufferingAndServletTimeoutOnlyForSse() throws Exception {
        var request = new MockHttpServletRequest("GET", "/api/ai/pipeline/reconnect");
        request.setServletPath("/api/ai/pipeline/reconnect");
        request.setAsyncSupported(true);
        var response = new MockHttpServletResponse();
        new AgentPipelineSseFilter().doFilter(request, response, (req, resp) -> req.startAsync());
        assertThat(response.getHeader("X-Accel-Buffering")).isEqualTo("no");
        assertThat(response.getHeader("Cache-Control")).contains("no-transform");
        assertThat(request.getAsyncContext().getTimeout()).isZero();
    }
}
