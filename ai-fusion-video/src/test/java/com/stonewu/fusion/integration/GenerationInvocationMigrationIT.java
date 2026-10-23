package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.Test;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@Testcontainers
class GenerationInvocationMigrationIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("generation_invocation_upgrade").withUsername("afv").withPassword("afv-test");
    private final ObjectMapper json = new ObjectMapper();
    private final AtomicLong taskIds = new AtomicLong(1);

    @Test
    void releasedUpgradeCreatesAgentIdentityConstraintsWithoutChangingRequestIds() throws Exception {
        var source = new DriverManagerDataSource(MYSQL.getJdbcUrl(), MYSQL.getUsername(), MYSQL.getPassword());
        var jdbc = new JdbcTemplate(source);
        Flyway.configure().dataSource(source).target("1.1.2.0.0").load().migrate();
        var history = jdbc.queryForList("SELECT version, checksum FROM flyway_schema_history ORDER BY installed_rank");
        var flyway = Flyway.configure().dataSource(source).load();
        flyway.migrate();
        flyway.validate();
        assertThat(jdbc.queryForList("SELECT version, checksum FROM flyway_schema_history ORDER BY installed_rank").subList(0, history.size())).isEqualTo(history);
        String context = json.writeValueAsString(Map.of("origin", "agent", "agentRunId", "original-run", "toolCallId", "原调用:\"\\\n" + "x".repeat(500)));
        String originalId = seed(jdbc, "tool:original-run:original-call", context, "a");
        String userId = seed(jdbc, "ordinary-user-request", "{\"origin\":\"user\"}", "a");
        assertThat(jdbc.queryForObject("SELECT request_id FROM afv_generation_submission WHERE id = ?", String.class, originalId))
                .isEqualTo("tool:original-run:original-call");
        assertThat(jdbc.queryForObject("SELECT agent_invocation_key FROM afv_generation_submission WHERE id = ?", String.class, originalId))
                .matches("[a-f0-9]{64}");
        assertThat(jdbc.queryForObject("SELECT agent_invocation_key FROM afv_generation_submission WHERE id = ?", String.class, userId)).isNull();
        assertThatThrownBy(() -> seed(jdbc, "new-request-same-call", context, "a")).isInstanceOf(DuplicateKeyException.class);
        seed(jdbc, "same-call-another-scope", context, "b");
        seed(jdbc, "other-user-request", "{\"origin\":\"user\"}", "a");
        seed(jdbc, "other-call", "{\"origin\":\"agent\",\"agentRunId\":\"original-run\",\"toolCallId\":\"another\"}", "a");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission", Integer.class)).isEqualTo(5);
    }

    private String seed(JdbcTemplate jdbc, String request, String context, String scope) {
        String id = UUID.randomUUID().toString();
        jdbc.update("""
                INSERT INTO afv_generation_submission(id, request_id, scope_key, request_hash, kind, task_id, public_task_id,
                    actor_user_id, context_json, model_snapshot_json, api_config_id, api_config_hash, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'IMAGE', ?, ?, 1, ?, '{}', 1, ?, 'UNKNOWN_SUBMISSION', UTC_TIMESTAMP(6), UTC_TIMESTAMP(6))
                """, id, request, scope.repeat(64), "c".repeat(64), taskIds.incrementAndGet(), id, context, "d".repeat(64));
        return id;
    }
}
