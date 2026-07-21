package com.stonewu.fusion.service.ai.agentscope.state;

import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.InMemoryAgentStateStore;
import io.agentscope.extensions.redis.state.RedisAgentStateStore;
import org.springframework.data.redis.core.StringRedisTemplate;

import java.util.Objects;

public final class AgentScopeStateStoreFactory {

    private final StateStoreFailureGuard failures;

    public AgentScopeStateStoreFactory(StateStoreFailureGuard failures) {
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
    }

    public AgentStateStore createInMemory() {
        return new FailClosedAgentStateStore(new InMemoryAgentStateStore(), failures);
    }

    public AgentStateStore createRedis(
            StringRedisTemplate redisTemplate,
            int maxConcurrentOperations,
            String keyPrefix) {
        Objects.requireNonNull(redisTemplate, "redisTemplate must not be null");
        if (keyPrefix == null || keyPrefix.isBlank()) {
            throw new IllegalArgumentException("keyPrefix must not be blank");
        }
        RedisAgentStateStore delegate = RedisAgentStateStore.builder()
                .clientAdapter(new SpringStringRedisClientAdapter(
                        redisTemplate, maxConcurrentOperations))
                .keyPrefix(keyPrefix.trim())
                .build();
        return new FailClosedAgentStateStore(delegate, failures);
    }
}
