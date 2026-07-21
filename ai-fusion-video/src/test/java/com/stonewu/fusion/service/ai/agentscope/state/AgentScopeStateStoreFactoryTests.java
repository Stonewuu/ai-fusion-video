package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.config.AgentScopeRuntimeConfiguration;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.InMemoryAgentStateStore;
import io.agentscope.extensions.redis.state.RedisAgentStateStore;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AgentScopeStateStoreFactoryTests {

    @Test
    void inMemoryModeSharesOneStorePerContextButNotAcrossContexts() {
        AtomicReference<InMemoryAgentStateStore> firstContextStore = new AtomicReference<>();

        contextRunner("in-memory").run(context -> {
            assertThat(context.getStartupFailure()).isNull();
            assertThat(context.getBeansOfType(AgentStateStore.class)).hasSize(1);
            AgentStateStore store = context.getBean(AgentStateStore.class);
            assertThat(store).isInstanceOf(FailClosedAgentStateStore.class);
            assertThat(delegate(store)).isInstanceOf(InMemoryAgentStateStore.class);
            firstContextStore.set((InMemoryAgentStateStore) delegate(store));
        });

        contextRunner("in-memory").run(context -> {
            assertThat(context.getStartupFailure()).isNull();
            assertThat(delegate(context.getBean(AgentStateStore.class))).isNotSameAs(firstContextStore.get());
        });

    }

    @Test
    void productionProfileBuildsOfficialRedisStoreWithExactPrefixAndNoFallback() {
        StringRedisTemplate redisTemplate = mock(StringRedisTemplate.class);
        when(redisTemplate.hasKey(anyString())).thenReturn(false);

        contextRunner("redis")
                .withPropertyValues("fusion.agentscope.v2.state.key-prefix=test:agentscope:v2:")
                .withBean(StringRedisTemplate.class, () -> redisTemplate)
                .run(context -> {
                    assertThat(context.getStartupFailure()).isNull();
                    assertThat(context.getBeansOfType(AgentStateStore.class)).hasSize(1);

                    AgentStateStore store = context.getBean(AgentStateStore.class);
                    assertThat(store).isInstanceOf(FailClosedAgentStateStore.class);
                    assertThat(delegate(store)).isInstanceOf(RedisAgentStateStore.class);
                    assertThat(store.exists("42", "afv:v2:conversation-7:assistant-v3")).isFalse();

                    verify(redisTemplate).hasKey(org.mockito.ArgumentMatchers.argThat(
                            key -> key.startsWith("test:agentscope:v2:")
                                    && key.contains("42")
                                    && key.contains("afv:v2:conversation-7:assistant-v3")));
                });
    }

    @Test
    void productionProfileFailsStartupWhenRedisDependencyIsMissing() {
        contextRunner("redis").run(context ->
                assertThat(context.getStartupFailure())
                        .isNotNull()
                        .hasMessageContaining("StringRedisTemplate"));
    }

    private ApplicationContextRunner contextRunner(String stateMode) {
        return new ApplicationContextRunner()
                .withPropertyValues(
                        "fusion.agentscope.v2.state.mode=" + stateMode,
                        "app.agentscope.runtime.state-threads=1",
                        "app.agentscope.runtime.journal-threads=1",
                        "app.agentscope.runtime.model-threads=1",
                        "app.agentscope.runtime.tool-threads=1")
                .withUserConfiguration(AgentScopeRuntimeConfiguration.class);
    }

    private AgentStateStore delegate(AgentStateStore store) {
        return (AgentStateStore) org.springframework.test.util.ReflectionTestUtils.getField(store, "delegate");
    }
}
