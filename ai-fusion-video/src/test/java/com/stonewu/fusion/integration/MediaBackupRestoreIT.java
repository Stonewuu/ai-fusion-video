package com.stonewu.fusion.integration;

import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.mapper.storage.StorageConfigMapper;
import com.stonewu.fusion.mapper.media.MediaResourceMapper;
import com.stonewu.fusion.mapper.media.MediaBlobMapper;
import com.stonewu.fusion.mapper.media.MediaEventMapper;
import com.stonewu.fusion.mapper.media.MediaPreviewJobMapper;
import com.stonewu.fusion.mapper.media.MediaSourceMapper;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.media.MediaBlobStore;
import com.stonewu.fusion.service.media.MediaCryptor;
import com.stonewu.fusion.service.media.lifecycle.MediaBlobLocks;
import com.stonewu.fusion.mapper.canvas.CanvasExportBlobMapper;
import org.springframework.integration.redis.util.RedisLockRegistry;
import com.stonewu.fusion.service.storage.S3ClientFactory;
import com.stonewu.fusion.service.storage.S3StorageConfigResolver;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.MountableFile;

import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * 在独立服务与目录恢复数据库和密文，不把导出 ZIP 当作灾备恢复。
 */
@Testcontainers
class MediaBackupRestoreIT extends MediaResourceTestContext {
    @Container
    static final MySQLContainer<?> RESTORED = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("media_resource_test").withUsername("afv").withPassword("afv-test");
    @TempDir
    Path backup;

    @Test
    void databaseKeysetsAndCiphertextTogetherRestoreMediaAndCanvasReferences() throws Exception {
        var media = upload(id(), png, "恢复后可读取.png", null, USER);
        var blob = repository.blob(repository.get(media.id(), false).blobId());
        String canvas = id(), node = id();
        jdbc.update("INSERT INTO afv_canvas(id,title,owner_type,owner_id,created_by) VALUES(?,'灾备恢复验收',1,71,71)",
                canvas);
        jdbc.update("""
                INSERT INTO afv_canvas_node(id,canvas_id,kind,title,x,y,config_json)
                VALUES(?,?,'text','保留的媒体引用',40,80,JSON_OBJECT('text',?))
                """, node, canvas, "[参考](media://" + media.id() + ")");
        var history = jdbc.queryForList("SELECT * FROM flyway_schema_history ORDER BY installed_rank");
        String nodeBefore = jdbc.queryForObject("SELECT config_json FROM afv_canvas_node WHERE id=?", String.class,
                node);

        var dump = MYSQL.execInContainer("mysqldump", "-uafv", "-pafv-test", "--single-transaction", "--no-tablespaces",
                "--hex-blob", "--routines", "--events", "--triggers", "--result-file=/tmp/g-backup.sql",
                MYSQL.getDatabaseName());
        assertThat(dump.getExitCode()).as(dump.getStderr()).isZero();
        Path databaseBackup = backup.resolve("database.sql");
        MYSQL.copyFileFromContainer("/tmp/g-backup.sql", databaseBackup.toString());
        RESTORED.copyFileToContainer(MountableFile.forHostPath(databaseBackup), "/tmp/g-backup.sql");
        var restore = RESTORED.execInContainer("sh", "-c",
                "mysql -uafv -pafv-test media_resource_test < /tmp/g-backup.sql");
        assertThat(restore.getExitCode()).as(restore.getStderr()).isZero();

        Path originalCiphertext = directory.resolve(".afv-managed").resolve(blob.objectKey());
        Path ciphertextBackup = backup.resolve("ciphertext.blob");
        Files.move(originalCiphertext, ciphertextBackup);
        Path restoredRoot = Files.createDirectory(backup.resolve("restored-storage"));
        Path restoredCiphertext = Files.createDirectory(restoredRoot.resolve(".afv-managed")).resolve(blob.objectKey());
        Files.copy(ciphertextBackup, restoredCiphertext);
        assertThat(originalCiphertext).doesNotExist();

        var source = new DriverManagerDataSource(RESTORED.getJdbcUrl(), RESTORED.getUsername(), RESTORED.getPassword());
        var restoredJdbc = new JdbcTemplate(source);
        Flyway.configure().dataSource(source).load().validate();
        assertThat(restoredJdbc.queryForList("SELECT * FROM flyway_schema_history ORDER BY installed_rank")).isEqualTo(
                history);
        assertThat(restoredJdbc.queryForObject("SELECT config_json FROM afv_canvas_node WHERE id=?", String.class,
                node)).isEqualTo(nodeBefore);
        restoredJdbc.update("UPDATE afv_storage_config SET base_path=? WHERE id=71", restoredRoot.toString());
        try (var restoredContext = new AnnotationConfigApplicationContext()) {
            var restoredSession = MybatisTestSupport.register(restoredContext, source);
            var restoredRepository = new MediaResourceRepository(restoredSession.getMapper(MediaResourceMapper.class),
                    restoredSession.getMapper(MediaBlobMapper.class), restoredSession.getMapper(MediaEventMapper.class),
                    restoredSession.getMapper(MediaPreviewJobMapper.class),
                    restoredSession.getMapper(MediaSourceMapper.class), new CanvasJson(mapper));
            var restoredBlob = restoredRepository.blob(restoredRepository.get(media.id(), false).blobId());
            assertThat(restoredRepository.retentionReference(media.id())).contains("CANVAS_NODE");
            var storage = new StorageConfig();
            storage.setId(71L);
            storage.setType("local");
            storage.setStatus(1);
            storage.setBasePath(
                    restoredJdbc.queryForObject("SELECT base_path FROM afv_storage_config WHERE id=71", String.class));
            var configurations = mock(StorageConfigMapper.class);
            when(configurations.selectById(71L)).thenReturn(storage);
            var cryptor = new MediaCryptor();
            var restoredStore = new MediaBlobStore(configurations, mock(S3StorageConfigResolver.class),
                    mock(S3ClientFactory.class),
                    cryptor, new MediaBlobLocks(restoredSession.getMapper(MediaBlobMapper.class),
                    restoredSession.getMapper(CanvasExportBlobMapper.class), context.getBean(RedisLockRegistry.class)));
            try (var channel = restoredStore.open(restoredBlob)) {
                assertThat(Channels.newInputStream(channel).readAllBytes()).isEqualTo(png);
            }
            assertThatThrownBy(() -> cryptor.decrypt(Files.newByteChannel(restoredCiphertext), cryptor.newKeyset(),
                    restoredBlob.resourceId()))
                    .isInstanceOf(IOException.class).hasMessage("媒体解密失败");
            assertThatThrownBy(
                    () -> cryptor.decrypt(Files.newByteChannel(restoredCiphertext), restoredBlob.keyset(), id()))
                    .isInstanceOf(IOException.class).hasMessage("媒体解密失败");

        }

        // 只输出不含密钥、密文和数据库内容的验收摘要；临时备份随测试目录清理。
        Files.writeString(Path.of("target/g-media-backup-restore-results.json"),
                mapper.writerWithDefaultPrettyPrinter().writeValueAsString(Map.of(
                        "database", MYSQL.getDockerImageName(), "separateMysqlInstance", true, "historyRowsPreserved",
                        history.size(),
                        "databaseBackupBytes", Files.size(databaseBackup), "ciphertextBytes",
                        Files.size(ciphertextBackup),
                        "originalCiphertextUnavailable", true, "restoredPlaintextMatches", true, "wrongKeyRejected",
                        true,
                        "wrongResourceRejected", true, "canvasReferencePreserved", true)));
    }
}
