package com.stonewu.fusion.integration;

import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.assertj.core.api.ThrowableAssert.ThrowingCallable;
import org.springframework.dao.DataAccessException;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import javax.sql.DataSource;
import java.sql.SQLException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@Testcontainers(disabledWithoutDocker = false)
class CanvasDocumentMigrationIT {
    @Container
    private static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("canvas_migration_test")
            .withUsername("afv")
            .withPassword("afv-test");

    @Container
    private static final MySQLContainer<?> FRESH = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("canvas_fresh_test").withUsername("afv").withPassword("afv-test");
    @TempDir static Path versionedMigrations;

    private static JdbcTemplate jdbc;
    private static Flyway flyway;
    private static List<Map<String, Object>> historicalChecksums;

    @BeforeAll
    static void upgradeExistingSchema() throws IOException {
        DataSource source = new DriverManagerDataSource(MYSQL.getJdbcUrl(), MYSQL.getUsername(), MYSQL.getPassword());
        jdbc = new JdbcTemplate(source);
        // 真实历史升级必须执行旧 V，不能被空数据库自动选用的 B 基线替代。
        try (var files = Files.list(Path.of("src/main/resources/db/migration"))) {
            for (Path file : files.filter(path -> path.getFileName().toString().matches("V.*\\.sql")).toList()) {
                Files.copy(file, versionedMigrations.resolve(file.getFileName()));
            }
        }
        Flyway.configure().dataSource(source).locations("filesystem:" + versionedMigrations.toAbsolutePath())
                .target("1.1.2.0.0").load().migrate();
        historicalChecksums = jdbc.queryForList("SELECT * FROM flyway_schema_history ORDER BY installed_rank");
        assertThat(historicalChecksums).hasSize(26);
        assertThat(historicalChecksums.getFirst()).containsEntry("script", "V1__init_mysql.sql");
        jdbc.update("INSERT INTO afv_project(name, owner_type, owner_id) VALUES (?, 1, 1)", "画布迁移前的项目");
        for (String table : List.of("afv_image_task", "afv_video_task")) {
            jdbc.update("INSERT INTO " + table + " (task_id,user_id,prompt,owner_type,owner_id,status) VALUES "
                    + "('migration-personal',71,'保留已有提示词',1,NULL,2),"
                    + "('migration-owned',71,'保留明确归属',1,72,2),"
                    + "('migration-team',71,'不推断团队归属',2,NULL,3)");
        }
        flyway = Flyway.configure().dataSource(source).load();
        flyway.migrate();
        flyway.validate();
    }

    @Test
    void validatesMigrationAndPreservesExistingProjectAndHistory() {
        assertThatCode(flyway::validate).doesNotThrowAnyException();
        assertThat(jdbc.queryForObject("SELECT name FROM afv_project WHERE name = ?", String.class, "画布迁移前的项目"))
                .isEqualTo("画布迁移前的项目");
        List<Map<String, Object>> after = jdbc.queryForList(
                "SELECT * FROM flyway_schema_history ORDER BY installed_rank");
        assertThat(after.subList(0, historicalChecksums.size())).containsExactlyElementsOf(historicalChecksums);
        assertThat(jdbc.queryForObject("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name IN (
                  'afv_canvas', 'afv_canvas_node', 'afv_canvas_edge', 'afv_canvas_member',
                  'afv_canvas_operation', 'afv_canvas_event', 'afv_canvas_revision', 'afv_canvas_comment')
                """, Integer.class)).isEqualTo(8);
    }

    @Test
    void backfillsOnlyKnownPersonalOwnersAndPreservesReleasedTaskContent() {
        for (String table : List.of("afv_image_task", "afv_video_task")) {
            assertThat(jdbc.queryForObject("SELECT owner_id FROM " + table + " WHERE task_id='migration-personal'", Long.class))
                    .isEqualTo(71L);
            assertThat(jdbc.queryForObject("SELECT owner_id FROM " + table + " WHERE task_id='migration-owned'", Long.class))
                    .isEqualTo(72L);
            assertThat(jdbc.queryForObject("SELECT owner_id FROM " + table + " WHERE task_id='migration-team'", Long.class))
                    .isNull();
            assertThat(jdbc.queryForObject("SELECT prompt FROM " + table + " WHERE task_id='migration-personal'", String.class))
                    .isEqualTo("保留已有提示词");
            assertThat(jdbc.queryForObject("SELECT status FROM " + table + " WHERE task_id='migration-personal'", Integer.class))
                    .isEqualTo(2);
            assertThat(jdbc.queryForObject("SELECT execution_context_json FROM " + table + " WHERE task_id='migration-personal'", String.class))
                    .isNull();
        }
    }

    @Test
    void freshBaselineAndFullVersionedUpgradeProduceTheSameColumnsAndIndexes() {
        var source = new DriverManagerDataSource(FRESH.getJdbcUrl(), FRESH.getUsername(), FRESH.getPassword());
        var freshFlyway = Flyway.configure().dataSource(source).load();
        freshFlyway.migrate();
        freshFlyway.validate();
        var fresh = new JdbcTemplate(source);
        assertThat(fresh.queryForObject("SELECT script FROM flyway_schema_history ORDER BY installed_rank LIMIT 1", String.class))
                .isEqualTo("B1.1.0.0.8__baseline_1.0.0.sql");
        String columns = """
                SELECT table_name,column_name,ordinal_position,column_type,is_nullable,column_default,extra,
                    character_set_name,collation_name,generation_expression
                FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name<>'flyway_schema_history'
                ORDER BY table_name,ordinal_position
                """;
        String indexes = """
                SELECT table_name,index_name,non_unique,seq_in_index,column_name,sub_part,index_type,is_visible,expression
                FROM information_schema.statistics WHERE table_schema=DATABASE() AND table_name<>'flyway_schema_history'
                ORDER BY table_name,index_name,seq_in_index
                """;
        assertThat(fresh.queryForList(columns)).containsExactlyElementsOf(jdbc.queryForList(columns));
        assertThat(fresh.queryForList(indexes)).containsExactlyElementsOf(jdbc.queryForList(indexes));
    }

    @Test
    void preventsCrossCanvasEdgesAndGroupingAtDatabaseBoundary() {
        String first = canvas();
        String second = canvas();
        String source = node(first);
        String target = node(second);
        assertThatThrownBy(() -> edge(first, source, target, 0)).isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> jdbc.update("UPDATE afv_canvas_node SET parent_id = ? WHERE id = ?", target, source))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void enforcesInputPositionUniquenessAndReleasesDeletedBindings() {
        String canvas = canvas();
        String source = node(canvas);
        String target = node(canvas);
        String original = edge(canvas, source, target, 0);
        assertThatThrownBy(() -> edge(canvas, source, target, 0)).isInstanceOf(DataIntegrityViolationException.class);
        jdbc.update("UPDATE afv_canvas_edge SET deleted = TRUE WHERE id = ?", original);
        assertThatCode(() -> edge(canvas, source, target, 0)).doesNotThrowAnyException();
    }

    @Test
    void enforcesCommandIdempotencyWithinUserAndCanvasScope() {
        String canvas = canvas();
        operation(canvas, "one-request", 1, 1);
        assertThatThrownBy(() -> operation(canvas, "one-request", 1, 2)).isInstanceOf(DataIntegrityViolationException.class);
        assertThatCode(() -> operation(canvas, "one-request", 2, 2)).doesNotThrowAnyException();
    }

    @Test
    void enforcesEventSequenceAndCommandOwnership() {
        String first = canvas();
        String second = canvas();
        String command = operation(first, "event-request", 1, 1);
        event(first, command, 1);
        assertThatThrownBy(() -> event(first, command, 1)).isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> event(second, command, 1)).isInstanceOf(DataIntegrityViolationException.class);
        rejectsCheck(() -> event(first, command, 0), "ck_canvas_event_sequence");
    }

    @Test
    void rejectsMalformedConfigurationAndIncompleteBusinessReferences() {
        String node = node(canvas());
        rejectsCheck(() -> jdbc.update("UPDATE afv_canvas_node SET config_json = '[]' WHERE id = ?", node),
                "ck_canvas_node_config");
        rejectsCheck(() -> jdbc.update("UPDATE afv_canvas_node SET resource_type = 'shot' WHERE id = ?", node),
                "ck_canvas_node_reference");
        rejectsCheck(() -> jdbc.update("UPDATE afv_canvas_node SET width = -1 WHERE id = ?", node),
                "ck_canvas_node_size");
    }

    private void rejectsCheck(ThrowingCallable operation, String constraint) {
        assertThatThrownBy(operation).isInstanceOfSatisfying(DataAccessException.class, failure -> {
            assertThat(failure.getMostSpecificCause()).isInstanceOfSatisfying(SQLException.class,
                    sql -> assertThat(sql.getErrorCode()).isEqualTo(3819));
            assertThat(failure.getMostSpecificCause().getMessage()).contains(constraint);
        });
    }

    private String canvas() {
        String id = UUID.randomUUID().toString();
        jdbc.update("INSERT INTO afv_canvas(id, title, owner_type, owner_id, created_by) VALUES (?, ?, 1, 1, 1)", id, "测试画布");
        return id;
    }

    private String node(String canvas) {
        String id = UUID.randomUUID().toString();
        jdbc.update("""
                INSERT INTO afv_canvas_node(id, canvas_id, kind, title, x, y, config_json)
                VALUES (?, ?, 'text', '文本', 0, 0, '{}')
                """, id, canvas);
        return id;
    }

    private String edge(String canvas, String source, String target, int index) {
        String id = UUID.randomUUID().toString();
        jdbc.update("""
                INSERT INTO afv_canvas_edge(id, canvas_id, kind, source_node_id, source_port, target_node_id, target_port, input_index)
                VALUES (?, ?, 'execution', ?, 'text', ?, 'texts', ?)
                """, id, canvas, source, target, index);
        return id;
    }

    private String operation(String canvas, String request, long userId, long revision) {
        String id = UUID.randomUUID().toString();
        jdbc.update("""
                INSERT INTO afv_canvas_operation(id, canvas_id, actor_user_id, request_id, request_hash,
                  origin, committed_revision, request_json, result_json)
                VALUES (?, ?, ?, ?, ?, 'user', ?, '{}', '{}')
                """, id, canvas, userId, request, "a".repeat(64), revision);
        return id;
    }

    private void event(String canvas, String command, long sequence) {
        jdbc.update("""
                INSERT INTO afv_canvas_event(id, canvas_id, sequence_no, operation_id, event_type, payload_json)
                VALUES (?, ?, ?, ?, 'canvas.changed', '{}')
                """, UUID.randomUUID().toString(), canvas, sequence, command);
    }
}
