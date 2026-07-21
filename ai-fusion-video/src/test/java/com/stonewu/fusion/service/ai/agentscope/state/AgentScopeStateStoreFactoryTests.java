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
    void localAndTestProfilesShareOneInMemoryStorePerContextButNotAcrossContexts() {
        AtomicReference<InMemoryAgentStateStore> firstContextStore = new AtomicReference<>();

        contextRunner("local").run(context -> {
            assertThat(context.getStartupFailure()).isNull();
            assertThat(context.getBeansOfType(AgentStateStore.class)).hasSize(1);
            AgentStateStore store = context.getBean(AgentStateStore.class);
            assertThat(store).isInstanceOf(FailClosedAgentStateStore.class);
            assertThat(delegate(store)).isInstanceOf(InMemoryAgentStateStore.class);
            firstContextStore.set((InMemoryAgentStateStore) delegate(store));
        });

        contextRunner("local").run(context -> {
            assertThat(context.getStartupFailure()).isNull();
            assertThat(delegate(context.getBean(AgentStateStore.class))).isNotSameAs(firstContextStore.get());
        });

        contextRunner("test").run(context -> {
            assertThat(context.getStartupFailure()).isNull();
            assertThat(context.getBeansOfType(AgentStateStore.class)).hasSize(1);
            assertThat(delegate(context.getBean(AgentStateStore.class)))
                    .isInstanceOf(InMemoryAgentStateStore.class);
        });
    }

    @Test
    void productionProfileBuildsOfficialRedisStoreWithExactPrefixAndNoFallback() {
        StringRedisTemplate redisTemplate = mock(StringRedisTemplate.class);
        when(redisTemplate.hasKey(anyString())).thenReturn(false);

        contextRunner("docker")
                .withBean(StringRedisTemplate.class, () -> redisTemplate)
                .run(context -> {
                    assertThat(context.getStartupFailure()).isNull();
                    assertThat(context.getBeansOfType(AgentStateStore.class)).hasSize(1);

                    AgentStateStore store = context.getBean(AgentStateStore.class);
                    assertThat(store).isInstanceOf(FailClosedAgentStateStore.class);
                    assertThat(delegate(store)).isInstanceOf(RedisAgentStateStore.class);
                    assertThat(store.exists("42", "afv:v2:conversation-7:assistant-v3")).isFalse();

                    verify(redisTemplate).hasKey(org.mockito.ArgumentMatchers.argThat(
                            key -> key.startsWith(AgentScopeStateStoreFactory.REDIS_KEY_PREFIX)
                                    && key.contains("42")
                                    && key.contains("afv:v2:conversation-7:assistant-v3")));
                });
    }

    @Test
    void productionProfileFailsStartupWhenRedisDependencyIsMissing() {
        contextRunner("docker").run(context ->
                assertThat(context.getStartupFailure())
                        .isNotNull()
                        .hasMessageContaining("StringRedisTemplate"));
    }

    private ApplicationContextRunner contextRunner(String profile) {
        return new ApplicationContextRunner()
                .withInitializer(context -> context.getEnvironment().setActiveProfiles(profile))
                .withPropertyValues(
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
