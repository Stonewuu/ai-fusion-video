package com.stonewu.fusion.integration;

import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import static org.assertj.core.api.Assertions.assertThat;

/** 正式版本升级必须保留已合成的视频和音轨，不能因为新增执行表而重新提交任务。 */
@Testcontainers
class EpisodeAudioMigrationIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("episode_audio_upgrade").withUsername("afv").withPassword("afv-test");

    @Test
    void upgradesReleasedEpisodeHistoryWithoutResubmittingOrChangingMedia() {
        var source = new DriverManagerDataSource(MYSQL.getJdbcUrl(), MYSQL.getUsername(), MYSQL.getPassword());
        var jdbc = new JdbcTemplate(source);
        Flyway.configure().dataSource(source).target("1.1.2.0.0").load().migrate();
        jdbc.update("""
                INSERT INTO afv_storyboard_episode(id,storyboard_id,title,composed_video_url,compose_status,
                    bgm_task_id,bgm_audio_url,bgm_sfx_audio_url,bgm_error_msg)
                VALUES (91,9,'已完成的分集','https://example.test/complete.mp4',2,
                    'released-audio','https://example.test/music.mp3','https://example.test/sfx.mp3',NULL),
                    (92,9,'配乐失败但视频已完成','https://example.test/video.mp4',2,
                    'released-failed',NULL,NULL,'原配乐处理失败')
                """);
        String content = """
                SELECT id,title,composed_video_url,compose_status,bgm_task_id,bgm_audio_url,bgm_sfx_audio_url,bgm_error_msg
                FROM afv_storyboard_episode ORDER BY id
                """;
        var before = jdbc.queryForList(content);

        var flyway = Flyway.configure().dataSource(source).load();
        flyway.migrate();
        flyway.validate();

        assertThat(jdbc.queryForList(content)).isEqualTo(before);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_storyboard_episode WHERE version=0 AND compose_job_id IS NULL", Integer.class))
                .isEqualTo(2);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_job", Integer.class)).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_episode_audio_job", Integer.class)).isZero();
    }
}
