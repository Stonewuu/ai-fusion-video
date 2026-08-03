package com.stonewu.fusion.integration;

import org.flywaydb.core.Flyway;
import org.flywaydb.core.api.FlywayException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ClassPathResource;
import org.springframework.dao.DataAccessException;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import javax.sql.DataSource;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.sql.SQLException;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@Testcontainers(disabledWithoutDocker = false)
class AgentPersistenceMigrationIT {

    private static final String PREVIOUS_VERSION = "1.0.6.1.4";

    @Container
    private static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video")
            .withUsername("afv")
            .withPassword("afv-test");

    @Container
    private static final MySQLContainer<?> UNSUPPORTED_MYSQL =
            new MySQLContainer<>("mysql:8.0.15")
                    .withDatabaseName("ai_fusion_video")
                    .withUsername("afv")
                    .withPassword("afv-test");

    private static JdbcTemplate jdbc;
    private static List<Long> legacyMessageIds;

    @BeforeAll
    static void migrateSchemaWithLegacyOrderingFixtures() {
        DataSource dataSource = new DriverManagerDataSource(
                MYSQL.getJdbcUrl(), MYSQL.getUsername(), MYSQL.getPassword());
        jdbc = new JdbcTemplate(dataSource);
        Flyway.configure()
                .dataSource(dataSource)
                .target(PREVIOUS_VERSION)
                .load()
                .migrate();
        seedLegacyMessageOrdering();
        Flyway.configure()
                .dataSource(dataSource)
                .load()
                .migrate();
    }

    @Test
    void createsRunEventUsageAndOrderingConstraints() {
        assertThat(tableCount(
                "afv_agent_workspace_config",
                "afv_agent_workspace_entry",
                "afv_agent_workspace_migration",
                "afv_agent_workspace_migration_item",
                "afv_agent_mcp_server"))
                .isEqualTo(5);
        assertThat(jdbc.queryForObject(
                "SELECT backend_type FROM afv_agent_workspace_config WHERE id = 1",
                String.class))
                .isEqualTo("database");
        assertThat(tableCount(
                "afv_agent_run", "afv_agent_event", "afv_agent_model_call_usage"))
                .isEqualTo(3);
        assertThat(indexColumns("afv_agent_message", "uk_agent_message_conv_order"))
                .containsExactly("conversation_id", "message_order");
        assertThat(indexColumns("afv_agent_message", "uk_agent_message_projection_key"))
                .containsExactly("projection_key");
        assertThat(indexColumns("afv_agent_message", "idx_agent_message_conv_run_order"))
                .containsExactly("conversation_id", "run_id", "message_order");
        assertThat(columnType("afv_agent_message", "message_order")).isEqualTo("bigint");
        assertThat(columnDefault("afv_agent_conversation", "next_message_order")).isEqualTo("1");
        assertThat(isNullable("afv_agent_conversation", "next_message_order")).isFalse();
        assertThat(columns("afv_agent_run"))
                .contains(
                        "parent_run_id",
                        "parent_tool_call_id",
                        "agent_name",
                        "deadline_at",
                        "projected_through_sequence",
                        "projection_completed_at");
        assertThat(isNullable("afv_agent_run", "deadline_at")).isFalse();
        assertThat(indexColumns("afv_agent_run", "uk_agent_run_parent_tool"))
                .containsExactly("parent_run_id", "parent_tool_call_id");
        assertThat(indexColumns("afv_agent_run", "idx_agent_run_parent_status"))
                .containsExactly("parent_run_id", "status", "id");
        assertThat(indexColumns("afv_agent_run", "idx_agent_run_status_deadline"))
                .containsExactly("status", "deadline_at", "id");
        assertThat(indexColumns("afv_agent_run", "uk_agent_run_active"))
                .containsExactly("active_conversation_id");
        assertThat(indexColumns("afv_agent_run", "idx_agent_run_lease"))
                .containsExactly("status", "lease_until");
        assertThat(indexColumns("afv_agent_run", "idx_agent_run_cancel"))
                .containsExactly("status", "cancel_next_attempt_at");
        assertThat(generationExpression("afv_agent_run", "active_conversation_id"))
                .contains("parent_run_id")
                .contains("conversation_id");
        assertThat(columns("afv_agent_event"))
                .contains(
                        "publish_required",
                        "publish_status",
                        "publish_claim_owner",
                        "publish_claim_until",
                        "next_publish_attempt_at");
        assertThat(indexColumns("afv_agent_event", "uk_agent_event_sequence"))
                .containsExactly("run_id", "sequence_no");
        assertThat(indexColumns("afv_agent_event", "idx_agent_event_publish"))
                .containsExactly("publish_status", "next_publish_attempt_at", "id");
        assertThat(columns("afv_agent_model_call_usage"))
                .contains(
                        "settlement_status",
                        "settlement_attempts",
                        "settlement_claim_owner",
                        "settlement_claim_until",
                        "downstream_settlement_id");
        assertThat(indexColumns("afv_agent_model_call_usage", "uk_agent_usage_call"))
                .containsExactly("run_id", "model_call_id");
        assertThat(indexColumns("afv_agent_model_call_usage", "idx_agent_usage_settlement"))
                .containsExactly("settlement_status", "next_settlement_attempt_at", "id");
        assertThat(columnCollation("afv_agent_run", "run_id")).isEqualTo("ascii_bin");
        assertThat(columnCollation("afv_agent_run", "parent_tool_call_id"))
                .isEqualTo("ascii_bin");
        assertThat(columnCollation("afv_agent_event", "run_id")).isEqualTo("ascii_bin");
        assertThat(columnCollation("afv_agent_model_call_usage", "model_call_id"))
                .isEqualTo("ascii_bin");

        assertLegacyMessagesWereStablyReordered();
        assertRootAndChildAdmissionConstraints();
        assertEventAndUsageIdempotencyConstraints();
        assertCheckConstraintsAreEnforced();
    }

    @Test
    void placesTheMysqlVersionGuardBeforeEverySchemaMutation() throws IOException {
        String migration = new ClassPathResource(
                        "db/migration/V1.0.6.1.5__agent_run_and_event.sql")
                .getContentAsString(StandardCharsets.UTF_8);
        int guard = migration.indexOf("PREPARE afv_mysql_guard");
        int firstMutation = migration.indexOf("CREATE TEMPORARY TABLE");

        assertThat(guard).isGreaterThanOrEqualTo(0);
        assertThat(firstMutation).isGreaterThan(guard);
        assertThat(migration)
                .contains("@afv_mysql_patch >= 16")
                .contains("VERSION() NOT LIKE '%MariaDB%'")
                .contains("__AGENTSCOPE_REQUIRES_MYSQL_8_0_16_OR_NEWER__");
    }

    @Test
    void rejectsMysqlBefore8016WithoutApplyingAnyTargetDdl() {
        DataSource unsupportedDataSource = new DriverManagerDataSource(
                UNSUPPORTED_MYSQL.getJdbcUrl(),
                UNSUPPORTED_MYSQL.getUsername(),
                UNSUPPORTED_MYSQL.getPassword());
        JdbcTemplate unsupportedJdbc = new JdbcTemplate(unsupportedDataSource);
        Flyway.configure()
                .dataSource(unsupportedDataSource)
                .target(PREVIOUS_VERSION)
                .load()
                .migrate();

        assertThatThrownBy(() -> Flyway.configure()
                        .dataSource(unsupportedDataSource)
                        .load()
                        .migrate())
                .isInstanceOf(FlywayException.class)
                .satisfies(failure -> {
                    Throwable rootCause = rootCause(failure);
                    assertThat(rootCause).isInstanceOf(SQLException.class);
                    assertThat(((SQLException) rootCause).getErrorCode()).isEqualTo(1146);
                    assertThat(rootCause.getMessage())
                            .contains("__AGENTSCOPE_REQUIRES_MYSQL_8_0_16_OR_NEWER__");
                });

        assertThat(tableCount(
                        unsupportedJdbc,
                        "afv_agent_run",
                        "afv_agent_event",
                        "afv_agent_model_call_usage"))
                .isZero();
        assertThat(columnCount(
                        unsupportedJdbc,
                        "afv_agent_conversation",
                        "next_message_order"))
                .isZero();
        assertThat(columnCount(
                        unsupportedJdbc,
                        "afv_agent_message",
                        "run_id"))
                .isZero();
    }

    private static void seedLegacyMessageOrdering() {
        jdbc.update("""
                INSERT INTO afv_agent_conversation(
                    conversation_id, user_id, title, message_count, deleted)
                VALUES ('legacy-ordering', 42, 'Legacy ordering', 999, 0)
                """);
        jdbc.update("""
                INSERT INTO afv_agent_conversation(
                    conversation_id, user_id, title, message_count, deleted)
                VALUES ('legacy-empty', 42, 'Legacy empty', 999, 0)
                """);
        jdbc.update("""
                INSERT INTO afv_agent_message(
                    conversation_id, role, content, message_order, deleted)
                VALUES ('legacy-ordering', 'assistant', 'order-four', 4, 0)
                """);
        jdbc.update("""
                INSERT INTO afv_agent_message(
                    conversation_id, role, content, message_order, deleted)
                VALUES ('legacy-ordering', 'user', 'order-two-first', 2, 0)
                """);
        jdbc.update("""
                INSERT INTO afv_agent_message(
                    conversation_id, role, content, message_order, deleted)
                VALUES ('legacy-ordering', 'tool', 'order-two-deleted', 2, 1)
                """);
        legacyMessageIds = jdbc.queryForList("""
                SELECT id
                FROM afv_agent_message
                WHERE conversation_id = 'legacy-ordering'
                ORDER BY id
                """, Long.class);
    }

    private static void assertLegacyMessagesWereStablyReordered() {
        List<OrderedMessage> messages = jdbc.query("""
                        SELECT content, message_order, deleted
                        FROM afv_agent_message
                        WHERE conversation_id = 'legacy-ordering'
                        ORDER BY message_order
                        """,
                (result, row) -> new OrderedMessage(
                        result.getString("content"),
                        result.getLong("message_order"),
                        result.getBoolean("deleted")));
        assertThat(messages)
                .containsExactly(
                        new OrderedMessage("order-two-first", 1, false),
                        new OrderedMessage("order-two-deleted", 2, true),
                        new OrderedMessage("order-four", 3, false));
        assertThat(jdbc.queryForList("""
                        SELECT id
                        FROM afv_agent_message
                        WHERE conversation_id = 'legacy-ordering'
                        ORDER BY id
                        """, Long.class))
                .containsExactlyElementsOf(legacyMessageIds);
        assertThat(jdbc.queryForObject("""
                        SELECT next_message_order
                        FROM afv_agent_conversation
                        WHERE conversation_id = 'legacy-ordering'
                        """, Long.class))
                .isEqualTo(4L);
        assertThat(jdbc.queryForObject("""
                        SELECT message_count
                        FROM afv_agent_conversation
                        WHERE conversation_id = 'legacy-ordering'
                        """, Integer.class))
                .isEqualTo(2);
        assertThat(jdbc.queryForMap("""
                        SELECT next_message_order, message_count
                        FROM afv_agent_conversation
                        WHERE conversation_id = 'legacy-empty'
                        """))
                .containsEntry("next_message_order", 1L)
                .containsEntry("message_count", 0);
        assertThatThrownBy(() -> jdbc.update("""
                        INSERT INTO afv_agent_message(
                            conversation_id, role, content, message_order, deleted)
                        VALUES ('legacy-ordering', 'user', 'duplicate-order', 3, 0)
                        """))
                .isInstanceOf(DuplicateKeyException.class);
    }

    private static void assertRootAndChildAdmissionConstraints() {
        insertRunningRoot("legacy-ordering", "root-1");
        assertThatCode(() -> insertRunningChild(
                        "legacy-ordering", "root-1", "tool-1", "child-a"))
                .doesNotThrowAnyException();
        assertThatThrownBy(() -> insertRunningChild(
                        "legacy-ordering", "root-1", "tool-1", "child-b"))
                .isInstanceOf(DuplicateKeyException.class);
        assertThatCode(() -> insertRunningChild(
                        "legacy-ordering", "root-1", "TOOL-1", "child-case-sensitive"))
                .doesNotThrowAnyException();
        assertThatThrownBy(() -> insertRunningRoot("legacy-ordering", "root-2"))
                .isInstanceOf(DuplicateKeyException.class);
        assertThat(jdbc.update("""
                UPDATE afv_agent_run
                SET status = 'COMPLETED', finished_at = CURRENT_TIMESTAMP(3)
                WHERE run_id = 'root-1'
                """)).isEqualTo(1);
        assertThatCode(() -> insertRunningRoot("legacy-ordering", "root-2"))
                .doesNotThrowAnyException();
    }

    private static void assertEventAndUsageIdempotencyConstraints() {
        insertEvent("root-1", 1);
        assertThatThrownBy(() -> insertEvent("root-1", 1))
                .isInstanceOf(DuplicateKeyException.class);
        insertUsage("root-1", "model-call-1");
        assertThatThrownBy(() -> insertUsage("root-1", "model-call-1"))
                .isInstanceOf(DuplicateKeyException.class);
    }

    private static void assertCheckConstraintsAreEnforced() {
        assertCheckViolation(() -> jdbc.update("""
                        INSERT INTO afv_agent_event(
                            run_id, sequence_no, raw_event_type, payload_json,
                            publish_required, publish_status)
                        VALUES ('root-1', 2, 'TEXT_BLOCK_DELTA', '{}', 0, 'PENDING')
                        """), "chk_agent_event_publish_state");
        assertCheckViolation(() -> jdbc.update("""
                        INSERT INTO afv_agent_event(
                            run_id, sequence_no, raw_event_type, payload_json,
                            publish_required, publish_status)
                        VALUES ('root-1', 2, 'TEXT_BLOCK_DELTA', '{}', 1, 'NOT_REQUIRED')
                        """), "chk_agent_event_publish_state");
        assertCheckViolation(() -> jdbc.update("""
                        INSERT INTO afv_agent_run(
                            run_id, conversation_id, user_id, kernel_fingerprint,
                            agent_definition_snapshot_json, agent_state_session_id,
                            status, deadline_at, started_at)
                        VALUES (
                            'invalid-status', 'other-conversation', 42, REPEAT('f', 64),
                            '{}', 'invalid-status-session', 'UNKNOWN',
                            DATE_ADD(CURRENT_TIMESTAMP(3), INTERVAL 10 MINUTE),
                            CURRENT_TIMESTAMP(3))
                        """), "chk_agent_run_status");
    }

    private static void assertCheckViolation(
            org.assertj.core.api.ThrowableAssert.ThrowingCallable operation,
            String constraintName) {
        assertThatThrownBy(operation)
                .isInstanceOf(DataAccessException.class)
                .satisfies(failure -> {
                    DataAccessException dataFailure = (DataAccessException) failure;
                    assertThat(dataFailure.getMostSpecificCause())
                            .isInstanceOf(SQLException.class);
                    SQLException sqlFailure =
                            (SQLException) dataFailure.getMostSpecificCause();
                    assertThat(sqlFailure.getErrorCode()).isEqualTo(3819);
                    assertThat(sqlFailure.getMessage()).contains(constraintName);
                });
    }

    private static void insertRunningRoot(String conversationId, String runId) {
        jdbc.update("""
                INSERT INTO afv_agent_run(
                    run_id, conversation_id, user_id, kernel_fingerprint,
                    agent_definition_snapshot_json, agent_state_session_id,
                    status, owner_instance_id, owner_epoch, lease_until,
                    deadline_at, started_at)
                VALUES (?, ?, 42, REPEAT('a', 64), '{}', ?, 'RUNNING',
                    'instance-a', 1,
                    DATE_ADD(CURRENT_TIMESTAMP(3), INTERVAL 20 SECOND),
                    DATE_ADD(CURRENT_TIMESTAMP(3), INTERVAL 10 MINUTE),
                    CURRENT_TIMESTAMP(3))
                """, runId, conversationId, runId + "-session");
    }

    private static void insertRunningChild(
            String conversationId,
            String parentRunId,
            String parentToolCallId,
            String childRunId) {
        jdbc.update("""
                INSERT INTO afv_agent_run(
                    run_id, conversation_id, user_id,
                    parent_run_id, parent_tool_call_id, agent_name,
                    kernel_fingerprint, agent_definition_snapshot_json,
                    agent_state_session_id, status, owner_instance_id,
                    owner_epoch, lease_until, deadline_at, started_at)
                VALUES (?, ?, 42, ?, ?, 'asset_image_gen', REPEAT('b', 64), '{}', ?,
                    'RUNNING', 'instance-a', 1,
                    DATE_ADD(CURRENT_TIMESTAMP(3), INTERVAL 20 SECOND),
                    DATE_ADD(CURRENT_TIMESTAMP(3), INTERVAL 10 MINUTE),
                    CURRENT_TIMESTAMP(3))
                """,
                childRunId,
                conversationId,
                parentRunId,
                parentToolCallId,
                childRunId + "-session");
    }

    private static void insertEvent(String runId, long sequence) {
        jdbc.update("""
                INSERT INTO afv_agent_event(
                    run_id, sequence_no, raw_event_type, payload_json,
                    publish_required, publish_status)
                VALUES (?, ?, 'TEXT_BLOCK_DELTA', '{}', 1, 'PENDING')
                """, runId, sequence);
    }

    private static void insertUsage(String runId, String modelCallId) {
        jdbc.update("""
                INSERT INTO afv_agent_model_call_usage(
                    run_id, model_call_id, provider, model_code, status,
                    settlement_status, started_at)
                VALUES (?, ?, 'openai', 'gpt-test', 'STARTED', 'PENDING', CURRENT_TIMESTAMP(3))
                """, runId, modelCallId);
    }

    private static int tableCount(String... tableNames) {
        return tableCount(jdbc, tableNames);
    }

    private static int tableCount(JdbcTemplate template, String... tableNames) {
        String placeholders = String.join(",", java.util.Collections.nCopies(tableNames.length, "?"));
        Object[] arguments = tableNames;
        Integer count = template.queryForObject("""
                        SELECT COUNT(*)
                        FROM information_schema.tables
                        WHERE table_schema = DATABASE()
                          AND table_name IN (""" + placeholders + ")",
                Integer.class,
                arguments);
        return count == null ? 0 : count;
    }

    private static int columnCount(
            JdbcTemplate template,
            String tableName,
            String columnName) {
        Integer count = template.queryForObject("""
                        SELECT COUNT(*)
                        FROM information_schema.columns
                        WHERE table_schema = DATABASE()
                          AND table_name = ? AND column_name = ?
                        """, Integer.class, tableName, columnName);
        return count == null ? 0 : count;
    }

    private static List<String> columns(String tableName) {
        return jdbc.queryForList("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = DATABASE() AND table_name = ?
                        ORDER BY ordinal_position
                        """, String.class, tableName);
    }

    private static String columnType(String tableName, String columnName) {
        return jdbc.queryForObject("""
                        SELECT data_type
                        FROM information_schema.columns
                        WHERE table_schema = DATABASE()
                          AND table_name = ? AND column_name = ?
                        """, String.class, tableName, columnName);
    }

    private static String columnDefault(String tableName, String columnName) {
        return jdbc.queryForObject("""
                        SELECT column_default
                        FROM information_schema.columns
                        WHERE table_schema = DATABASE()
                          AND table_name = ? AND column_name = ?
                        """, String.class, tableName, columnName);
    }

    private static String columnCollation(String tableName, String columnName) {
        return jdbc.queryForObject("""
                        SELECT collation_name
                        FROM information_schema.columns
                        WHERE table_schema = DATABASE()
                          AND table_name = ? AND column_name = ?
                        """, String.class, tableName, columnName);
    }

    private static String generationExpression(String tableName, String columnName) {
        return jdbc.queryForObject("""
                        SELECT generation_expression
                        FROM information_schema.columns
                        WHERE table_schema = DATABASE()
                          AND table_name = ? AND column_name = ?
                        """, String.class, tableName, columnName);
    }

    private static boolean isNullable(String tableName, String columnName) {
        return "YES".equals(jdbc.queryForObject("""
                        SELECT is_nullable
                        FROM information_schema.columns
                        WHERE table_schema = DATABASE()
                          AND table_name = ? AND column_name = ?
                        """, String.class, tableName, columnName));
    }

    private static List<String> indexColumns(String tableName, String indexName) {
        return jdbc.queryForList("""
                        SELECT column_name
                        FROM information_schema.statistics
                        WHERE table_schema = DATABASE()
                          AND table_name = ? AND index_name = ?
                        ORDER BY seq_in_index
                        """, String.class, tableName, indexName);
    }

    private static Throwable rootCause(Throwable failure) {
        Throwable current = failure;
        while (current.getCause() != null && current.getCause() != current) {
            current = current.getCause();
        }
        return current;
    }

    private record OrderedMessage(String content, long order, boolean deleted) {
    }
}
