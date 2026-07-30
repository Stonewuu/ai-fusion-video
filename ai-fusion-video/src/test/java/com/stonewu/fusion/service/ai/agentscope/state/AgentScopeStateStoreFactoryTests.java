package com.stonewu.fusion.service.ai.agentscope.state;

import com.stonewu.fusion.config.AgentScopeRuntimeConfiguration;
import io.agentscope.core.state.AgentStateStore;
import io.agentscope.core.state.InMemoryAgentStateStore;
import io.agentscope.extensions.mysql.state.MysqlAgentStateStore;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
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
    void productionProfileBuildsOfficialMysqlStoreWithNoFallback() throws Exception {
        DataSource dataSource = mock(DataSource.class);
        Connection connection = mock(Connection.class);
        PreparedStatement statement = mock(PreparedStatement.class);
        ResultSet resultSet = mock(ResultSet.class);
        when(dataSource.getConnection()).thenReturn(connection);
        when(connection.prepareStatement(org.mockito.ArgumentMatchers.anyString()))
                .thenReturn(statement);
        when(statement.executeQuery()).thenReturn(resultSet);
        when(resultSet.next()).thenReturn(true);

        contextRunner("mysql")
                .withPropertyValues(
                        "fusion.agentscope.v2.state.database-name=ai_fusion_video",
                        "fusion.agentscope.v2.state.table-name=afv_agent_state")
                .withBean(DataSource.class, () -> dataSource)
                .run(context -> {
                    assertThat(context.getStartupFailure()).isNull();
                    assertThat(context.getBeansOfType(AgentStateStore.class)).hasSize(1);

                    AgentStateStore store = context.getBean(AgentStateStore.class);
                    assertThat(store).isInstanceOf(FailClosedAgentStateStore.class);
                    assertThat(delegate(store))
                            .isInstanceOfSatisfying(MysqlAgentStateStore.class, mysql -> {
                                assertThat(mysql.getDatabaseName()).isEqualTo("ai_fusion_video");
                                assertThat(mysql.getTableName()).isEqualTo("afv_agent_state");
                            });
                });
    }

    @Test
    void productionProfileFailsStartupWhenDataSourceIsMissing() {
        contextRunner("mysql").run(context ->
                assertThat(context.getStartupFailure())
                        .isNotNull()
                        .hasMessageContaining("DataSource"));
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
