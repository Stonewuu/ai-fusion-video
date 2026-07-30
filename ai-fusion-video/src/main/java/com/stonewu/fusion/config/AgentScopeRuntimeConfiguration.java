package com.stonewu.fusion.config;

import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.state.AgentScopeStateStoreFactory;
import com.stonewu.fusion.service.ai.agentscope.state.AgentStatePreflight;
import com.stonewu.fusion.service.ai.agentscope.state.InMemoryStateStoreFailureGuard;
import com.stonewu.fusion.service.ai.agentscope.state.StateStoreFailureGuard;
import com.stonewu.fusion.service.ai.run.AgentRuntimeMetrics;
import io.agentscope.core.state.AgentStateStore;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

import javax.sql.DataSource;

@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties({AgentScopeRuntimeProperties.class, AgentScopeV2Properties.class})
public class AgentScopeRuntimeConfiguration {

    @Bean
    @ConditionalOnMissingBean(AgentRuntimeMetrics.class)
    public AgentRuntimeMetrics agentRuntimeMetrics(
            ObjectProvider<io.micrometer.core.instrument.MeterRegistry> registries) {
        return new AgentRuntimeMetrics(registries);
    }

    @Bean(destroyMethod = "close")
    public AgentRuntimeSchedulers agentRuntimeSchedulers(
            AgentScopeRuntimeProperties properties,
            AgentRuntimeMetrics metrics) {
        return new AgentRuntimeSchedulers(properties, metrics);
    }

    @Bean
    public StateStoreFailureGuard stateStoreFailureGuard() {
        return new InMemoryStateStoreFailureGuard();
    }

    @Bean
    public AgentScopeStateStoreFactory agentScopeStateStoreFactory(
            StateStoreFailureGuard failures,
            AgentRuntimeMetrics metrics) {
        return new AgentScopeStateStoreFactory(failures, metrics);
    }

    @Bean(destroyMethod = "close")
    @Primary
    public AgentStateStore agentScopeStateStore(
            AgentScopeStateStoreFactory factory,
            ObjectProvider<DataSource> dataSourceProvider,
            AgentScopeV2Properties v2Properties) {
        AgentScopeV2Properties.State state = v2Properties.getState();
        if (state.getMode() == AgentScopeV2Properties.Mode.IN_MEMORY) {
            return factory.createInMemory();
        }
        DataSource dataSource = dataSourceProvider.getIfAvailable();
        if (dataSource == null) {
            throw new IllegalStateException(
                    "DataSource is required when fusion.agentscope.v2.state.mode=mysql");
        }
        return factory.createMysql(
                dataSource, state.getDatabaseName(), state.getTableName());
    }

    @Bean
    public AgentStatePreflight agentStatePreflight(
            AgentStateStore store,
            StateStoreFailureGuard failures,
            AgentRuntimeSchedulers schedulers) {
        return new AgentStatePreflight(store, failures, schedulers);
    }
}
