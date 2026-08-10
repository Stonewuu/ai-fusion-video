package com.stonewu.fusion.service.ai.agentscope.state;

import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.InMemoryAgentStateStore;
import io.agentscope.extensions.mysql.state.MysqlAgentStateStore;
import com.stonewu.fusion.service.ai.run.AgentRuntimeMetrics;

import javax.sql.DataSource;
import java.util.Objects;

public final class AgentScopeStateStoreFactory {

    private final StateStoreFailureGuard failures;
    private final AgentRuntimeMetrics metrics;

    public AgentScopeStateStoreFactory(StateStoreFailureGuard failures) {
        this(failures, AgentRuntimeMetrics.noop());
    }

    public AgentScopeStateStoreFactory(
            StateStoreFailureGuard failures, AgentRuntimeMetrics metrics) {
        this.failures = Objects.requireNonNull(failures, "failures must not be null");
        this.metrics = Objects.requireNonNull(metrics, "metrics must not be null");
    }

    public AgentStateStore createInMemory() {
        return new FailClosedAgentStateStore(
                new InMemoryAgentStateStore(), failures, metrics);
    }

    public AgentStateStore createMysql(
            DataSource dataSource, String databaseName, String tableName) {
        MysqlAgentStateStore delegate = new MysqlAgentStateStore(
                Objects.requireNonNull(dataSource, "dataSource must not be null"),
                databaseName,
                tableName,
                false);
        return new FailClosedAgentStateStore(delegate, failures, metrics);
    }
}
