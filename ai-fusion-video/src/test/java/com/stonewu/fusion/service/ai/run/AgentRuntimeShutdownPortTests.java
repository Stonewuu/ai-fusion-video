package com.stonewu.fusion.service.ai.run;

import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

import static org.assertj.core.api.Assertions.assertThat;

class AgentRuntimeShutdownPortTests {

    @Test
    void exposesOnlyTheKernelOwnedReactiveShutdownBoundary() throws Exception {
        Method shutdown = AgentRuntimeShutdownPort.class.getDeclaredMethod("shutdown", Duration.class);
        assertThat(shutdown.getReturnType()).isEqualTo(Mono.class);
        assertThat(AgentRuntimeShutdownPort.class.getDeclaredMethods()).containsExactly(shutdown);

        Path sourceRoot = Path.of("src/main/java/com/stonewu/fusion/service/ai/run");
        assertThat(Files.exists(sourceRoot.resolve("AgentRuntimeShutdownPort.java"))).isTrue();
        assertThat(Files.exists(sourceRoot.resolve("AgentScopeKernelLifecycle.java"))).isTrue();
    }
}
