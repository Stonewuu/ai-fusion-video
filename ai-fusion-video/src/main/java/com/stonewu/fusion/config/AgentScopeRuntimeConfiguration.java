package com.stonewu.fusion.config;

import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolkitResources;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelToolRegistry;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.AgentScopeStateStoreFactory;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStatePreflight;
import com.stonewu.fusion.service.ai.agentscope.state.InMemoryStateStoreFailureGuard;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreFailureGuard;
import io.agentscope.core.state.AgentStateStore;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.context.annotation.Profile;
import org.springframework.data.redis.core.StringRedisTemplate;

@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties(AgentScopeRuntimeProperties.class)
public class AgentScopeRuntimeConfiguration {

    @Bean(destroyMethod = "close")
    public AgentRuntimeSchedulers agentRuntimeSchedulers(AgentScopeRuntimeProperties properties) {
        return new AgentRuntimeSchedulers(properties);
    }

    @Bean
    public StateStoreFailureGuard stateStoreFailureGuard() {
        return new InMemoryStateStoreFailureGuard();
    }

    @Bean
    @ConditionalOnMissingBean(AgentKernelToolRegistry.class)
    public AgentKernelToolRegistry agentKernelToolRegistry() {
        return (spec, toolkit) -> AgentKernelToolkitResources.none();
    }

    @Bean
    public AgentScopeStateStoreFactory agentScopeStateStoreFactory(
            StateStoreFailureGuard failures) {
        return new AgentScopeStateStoreFactory(failures);
    }

    @Bean(destroyMethod = "close")
    @Primary
    @Profile({"local", "test"})
    public AgentStateStore localAgentScopeStateStore(
            AgentScopeStateStoreFactory factory) {
        return factory.createInMemory();
    }

    @Bean(destroyMethod = "close")
    @Primary
    @Profile("!local & !test")
    public AgentStateStore redisAgentScopeStateStore(
            AgentScopeStateStoreFactory factory,
            StringRedisTemplate redisTemplate,
            AgentScopeRuntimeProperties properties) {
        return factory.createRedis(redisTemplate, properties.getStateThreads());
    }

    @Bean
    public AgentStatePreflight agentStatePreflight(
            AgentStateStore store,
            StateStoreFailureGuard failures,
            AgentRuntimeSchedulers schedulers) {
        return new AgentStatePreflight(store, failures, schedulers);
    }
}
