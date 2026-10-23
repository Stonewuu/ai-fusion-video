package com.stonewu.fusion.integration;

import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.Test;
import org.springframework.jdbc.UncategorizedSQLException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** 合并建表后，画布复制和模板导入仍须遵守各自的来源约束。 */
@Testcontainers
class CanvasTemplateSourceMigrationIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("canvas_template_upgrade").withUsername("afv").withPassword("afv-test");

    @Test
    void enforcesCanvasAndTemplateOriginsAfterReleasedVersionUpgrade() {
        var source = new DriverManagerDataSource(MYSQL.getJdbcUrl(), MYSQL.getUsername(), MYSQL.getPassword());
        var jdbc = new JdbcTemplate(source);
        Flyway.configure().dataSource(source).target("1.1.2.0.0").load().migrate();
        var history = jdbc.queryForList("SELECT version,script,checksum FROM flyway_schema_history ORDER BY installed_rank");
        var flyway = Flyway.configure().dataSource(source).load();
        flyway.migrate();
        flyway.validate();
        assertThat(jdbc.queryForList("SELECT version,script,checksum FROM flyway_schema_history ORDER BY installed_rank")
                .subList(0, history.size())).isEqualTo(history);

        String canvas = UUID.randomUUID().toString();
        String node = UUID.randomUUID().toString();
        String snapshot = UUID.randomUUID().toString();
        jdbc.update("INSERT INTO afv_canvas(id,title,owner_type,owner_id,created_by) VALUES(?,'固定结果',1,1,1)", canvas);
        jdbc.update("""
                INSERT INTO afv_canvas_value_snapshot(id,canvas_id,node_id,port_id,schema_version,value_json,
                    source_canvas_id,source_output_id,created_by,created_at)
                VALUES(?,?,?,'text',1,JSON_OBJECT('id','original','type','text','text','原正文'),?,'original',1,UTC_TIMESTAMP(6))
                """, snapshot, canvas, node, canvas);
        String before = jdbc.queryForObject("SELECT value_json FROM afv_canvas_value_snapshot WHERE id=?", String.class, snapshot);
        assertThat(jdbc.queryForObject("SELECT source_kind FROM afv_canvas_value_snapshot WHERE id=?", String.class, snapshot))
                .isEqualTo("CANVAS");
        assertThat(jdbc.queryForObject("SELECT source_canvas_id FROM afv_canvas_value_snapshot WHERE id=?", String.class, snapshot))
                .isEqualTo(canvas);
        assertThatThrownBy(() -> jdbc.update("UPDATE afv_canvas_value_snapshot SET source_kind='TEMPLATE' WHERE id=?", snapshot))
                .isInstanceOfSatisfying(UncategorizedSQLException.class,
                        failure -> assertThat(failure.getSQLException().getErrorCode()).isEqualTo(3819));
        assertThat(jdbc.queryForObject("SELECT value_json FROM afv_canvas_value_snapshot WHERE id=?", String.class, snapshot))
                .isEqualTo(before);

        String imported = UUID.randomUUID().toString();
        jdbc.update("""
                INSERT INTO afv_canvas_value_snapshot(id,canvas_id,node_id,port_id,schema_version,value_json,
                    source_output_id,created_by,created_at,source_kind,template_hash)
                VALUES(?,?,?,'text',1,JSON_OBJECT('id','imported','type','text','text','导入正文'),'imported',1,UTC_TIMESTAMP(6),'TEMPLATE',?)
                """, imported, canvas, node, "a".repeat(64));
        assertThat(jdbc.queryForObject("SELECT source_canvas_id FROM afv_canvas_value_snapshot WHERE id=?", String.class, imported))
                .isNull();
        assertThatThrownBy(() -> jdbc.update("UPDATE afv_canvas_value_snapshot SET template_hash='invalid' WHERE id=?", imported))
                .isInstanceOfSatisfying(UncategorizedSQLException.class,
                        failure -> assertThat(failure.getSQLException().getErrorCode()).isEqualTo(3819));
    }
}
