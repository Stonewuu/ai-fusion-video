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
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.data.redis.core.StringRedisTemplate;

@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties({AgentScopeRuntimeProperties.class, AgentScopeV2Properties.class})
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
    public AgentStateStore agentScopeStateStore(
            AgentScopeStateStoreFactory factory,
            ObjectProvider<StringRedisTemplate> redisTemplateProvider,
            AgentScopeRuntimeProperties runtimeProperties,
            AgentScopeV2Properties v2Properties) {
        AgentScopeV2Properties.State state = v2Properties.getState();
        if (state.getMode() == AgentScopeV2Properties.Mode.IN_MEMORY) {
            return factory.createInMemory();
        }
        StringRedisTemplate redisTemplate = redisTemplateProvider.getIfAvailable();
        if (redisTemplate == null) {
            throw new IllegalStateException(
                    "StringRedisTemplate is required when fusion.agentscope.v2.state.mode=redis");
        }
        return factory.createRedis(
                redisTemplate, runtimeProperties.getStateThreads(), state.getKeyPrefix());
    }

    @Bean
    public AgentStatePreflight agentStatePreflight(
            AgentStateStore store,
            StateStoreFailureGuard failures,
            AgentRuntimeSchedulers schedulers) {
        return new AgentStatePreflight(store, failures, schedulers);
    }
}
