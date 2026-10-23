package com.stonewu.fusion.integration;

import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.generation.GenerationTasks;
import com.stonewu.fusion.mapper.media.MediaResourceMapper;
import com.stonewu.fusion.mapper.media.MediaBlobMapper;
import com.stonewu.fusion.mapper.media.MediaEventMapper;
import com.stonewu.fusion.mapper.media.MediaPreviewJobMapper;
import com.stonewu.fusion.mapper.media.MediaSourceMapper;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.CacheConfig;
import com.stonewu.fusion.controller.media.MediaContentController;
import com.stonewu.fusion.controller.media.MediaExceptionHandler;
import com.stonewu.fusion.controller.media.MediaResourceController;
import com.stonewu.fusion.entity.storage.StorageConfig;
import com.stonewu.fusion.mapper.storage.StorageConfigMapper;
import com.stonewu.fusion.repository.canvas.CanvasAccessRepository;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import com.stonewu.fusion.service.media.MediaSourceQueries;
import com.stonewu.fusion.service.media.MediaSourceFetcher;
import com.stonewu.fusion.service.media.MediaSourceImportService;
import com.stonewu.fusion.service.media.MediaRemoteAddressPolicy;
import com.stonewu.fusion.service.storage.LegacyMediaRoot;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.MediaBusinessReferences;
import com.stonewu.fusion.service.media.GenerationMediaSourceQueries;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.media.MediaBindingObserver;
import com.stonewu.fusion.service.media.ResourceBindMediaHandler;
import com.stonewu.fusion.service.media.MediaContentReader;
import com.stonewu.fusion.controller.media.MediaBindingController;
import com.stonewu.fusion.repository.resource.ResourceCommandRepository;
import com.stonewu.fusion.repository.resource.ResourceRevisionRepository;
import com.stonewu.fusion.repository.resource.AssetMutationRepository;
import com.stonewu.fusion.repository.resource.StoryboardMutationRepository;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.AssetCommandSupport;
import com.stonewu.fusion.service.resource.AssetFields;
import com.stonewu.fusion.service.resource.AssetResourcePatchHandler;
import com.stonewu.fusion.service.resource.AssetResourceCreateHandler;
import com.stonewu.fusion.service.resource.AssetResourceDeleteHandler;
import com.stonewu.fusion.service.resource.ProjectFields;
import com.stonewu.fusion.service.resource.ProjectResourcePatchHandler;
import com.stonewu.fusion.service.resource.StoryboardCommandSupport;
import com.stonewu.fusion.service.resource.StoryboardFields;
import com.stonewu.fusion.service.resource.StoryboardRelations;
import com.stonewu.fusion.service.resource.StoryboardVersionView;
import com.stonewu.fusion.service.resource.StoryboardResourcePatchHandler;
import com.stonewu.fusion.service.canvas.CanvasRequestValidator;
import com.stonewu.fusion.service.canvas.CanvasBindingPolicy;
import com.stonewu.fusion.service.canvas.CanvasCommandService;
import com.stonewu.fusion.service.canvas.CanvasGraphValidator;
import com.stonewu.fusion.service.canvas.CanvasMediaGraphPolicy;
import com.stonewu.fusion.service.canvas.command.CanvasEditEngine;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasConfigValidator;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.service.generation.job.GenerationOutbox;
import com.stonewu.fusion.service.media.importing.*;
import com.stonewu.fusion.service.media.preview.*;
import com.stonewu.fusion.controller.media.MediaPreviewController;
import com.stonewu.fusion.service.media.job.MediaProcessRunner;
import com.stonewu.fusion.controller.media.MediaImportController;
import org.springframework.transaction.support.TransactionTemplate;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import jakarta.validation.ValidatorFactory;
import com.stonewu.fusion.service.media.MediaAccessTickets;
import com.stonewu.fusion.service.media.MediaBlobStore;
import com.stonewu.fusion.service.media.lifecycle.MediaBlobLocks;
import com.stonewu.fusion.service.media.lifecycle.MediaBlobLockConfiguration;
import com.stonewu.fusion.service.media.MediaCryptor;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaMetadataCache;
import com.stonewu.fusion.service.media.MediaProbe;
import com.stonewu.fusion.service.media.MediaPresetReferences;
import com.stonewu.fusion.service.media.MediaRegistryTransactions;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.MediaStorageConfigGuard;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.storage.S3ClientFactory;
import com.stonewu.fusion.service.storage.S3StorageConfigResolver;
import com.stonewu.fusion.service.storage.StorageProviderRegistry;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.jdbc.core.JdbcTemplate;
import com.stonewu.fusion.service.media.job.MediaJobFence;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.media.job.MediaJobNotifications;
import com.stonewu.fusion.service.resource.ComposeWritebackHandler;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.junit.jupiter.Container;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;
import static org.mockito.AdditionalAnswers.delegatesTo;

abstract class MediaResourceTestContext {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("media_resource_test").withUsername("afv").withPassword("afv-test");
    @Container
    static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    @Container
    static final GenericContainer<?> S3 = new GenericContainer<>("quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z")
            .withEnv("MINIO_ROOT_USER", "media-test").withEnv("MINIO_ROOT_PASSWORD", "media-test-password")
            .withCommand("server /data").withExposedPorts(9000)
            .waitingFor(Wait.forHttp("/minio/health/ready").forPort(9000));
    static final CanvasActor OWNER = new CanvasActor(71, null);
    static final CanvasActor OTHER = new CanvasActor(72, null);
    static final ResourceCommandContext USER = ResourceCommandContext.user(OWNER, null);
    static AnnotationConfigApplicationContext context;
    static JdbcTemplate jdbc;
    static ObjectMapper mapper;
    static MediaResourceRepository repository;
    static MediaIntakeService intake;
    static MediaRegistryTransactions registry;
    static MediaResourceQueries queries;
    static MediaAccessTickets tickets;
    static MediaBlobStore blobs;
    static StorageConfigMapper storageMapper;
    static CanvasActorResolver actors;
    static Path directory;
    static byte[] png;
    static MockMvc http;
    static Flyway flyway;

    @Configuration
    @EnableTransactionManagement
    static class Transactions {
    }

    @BeforeAll
    static void start() throws Exception {
        directory = Files.createTempDirectory("afv-media-it-");
        var source = new DriverManagerDataSource(MYSQL.getJdbcUrl(), MYSQL.getUsername(), MYSQL.getPassword());
        jdbc = new JdbcTemplate(source);
        Flyway.configure().dataSource(source).target("1.1.2.0.0").load().migrate();
        List<Map<String, Object>> history = jdbc.queryForList(
                "SELECT version, checksum FROM flyway_schema_history ORDER BY installed_rank");
        flyway = Flyway.configure().dataSource(source).load();
        flyway.migrate();
        flyway.validate();
        assertThat(jdbc.queryForList("SELECT version, checksum FROM flyway_schema_history ORDER BY installed_rank")
                .subList(0, history.size())).isEqualTo(history);
        jdbc.update(
                "INSERT INTO sys_user(id, username, password) VALUES (71, 'media-owner', 'test'), (72, 'media-other', 'test')");
        jdbc.update("INSERT INTO afv_project(id, name, owner_type, owner_id) VALUES (71, '媒体项目', 1, 71)");
        jdbc.update(
                "INSERT INTO afv_storage_config(id, name, type, base_path, is_default) VALUES (71, '本地测试', 'local', ?, TRUE)",
                directory.toString());
        jdbc.update("""
                INSERT INTO afv_storage_config(id, name, type, provider, endpoint, bucket_name, access_key, secret_key, region, base_path)
                VALUES (72, '对象测试', 's3', 'minio', ?, 'media-test', 'media-test', 'media-test-password', 'us-east-1', 'test')
                """, "http://" + S3.getHost() + ":" + S3.getMappedPort(9000));
        mapper = new ObjectMapper().findAndRegisterModules();
        context = new AnnotationConfigApplicationContext();
        var session = MybatisTestSupport.register(context, source);
        repository = spy(new MediaResourceRepository(session.getMapper(MediaResourceMapper.class),
                session.getMapper(MediaBlobMapper.class), session.getMapper(MediaEventMapper.class),
                session.getMapper(MediaPreviewJobMapper.class), session.getMapper(MediaSourceMapper.class),
                new CanvasJson(mapper)));
        storageMapper = mock(StorageConfigMapper.class, delegatesTo(session.getMapper(StorageConfigMapper.class)));
        actors = mock(CanvasActorResolver.class);
        when(actors.current()).thenReturn(OWNER);
        context.getBeanFactory().registerSingleton("mediaJobNotifications", mock(MediaJobNotifications.class));
        context.registerBean(JdbcTemplate.class, () -> jdbc);
        context.registerBean(PlatformTransactionManager.class, () -> new DataSourceTransactionManager(source));
        context.registerBean(TransactionTemplate.class,
                () -> new TransactionTemplate(context.getBean(PlatformTransactionManager.class)));
        context.registerBean(ObjectMapper.class, () -> mapper);
        context.registerBean(ValidatorFactory.class, Validation::buildDefaultValidatorFactory);
        context.registerBean(Validator.class, () -> context.getBean(ValidatorFactory.class).getValidator());
        context.registerBean(StorageConfigMapper.class, () -> storageMapper);
        context.registerBean(MediaResourceRepository.class, () -> repository);
        context.registerBean(MediaCryptor.class, () -> spy(new MediaCryptor()));
        context.registerBean(S3ClientFactory.class, () -> spy(new S3ClientFactory()));
        context.registerBean(CanvasActorResolver.class, () -> actors);
        context.registerBean(RedisConnectionFactory.class,
                () -> new LettuceConnectionFactory(REDIS.getHost(), REDIS.getMappedPort(6379)));
        context.register(Transactions.class, CacheConfig.class, CanvasJson.class, CanvasDocumentRepository.class,
                CanvasGraphRepository.class, CanvasJournalRepository.class, CanvasBindingPolicy.class,
                CanvasCommandService.class,
                CanvasGraphValidator.class, CanvasEditEngine.class, CanvasCapabilityRegistry.class,
                CanvasConfigValidator.class, CanvasMediaGraphPolicy.class,
                MediaPreviewOwnership.class, MediaPreviewQueries.class, MediaPreviewService.class,
                MediaPreviewJobs.class,
                MediaPreviewIntake.class, MediaPreviewWorker.class, MediaPreviewController.class,
                MediaProcessRunner.class,
                GenerationOutbox.class, MediaImportAccess.class, MediaImportQueries.class, MediaImportService.class,
                MediaImportPlacementService.class,
                MediaImportJobs.class, MediaImportWorker.class, MediaImportController.class,
                CanvasRequestValidator.class, ResourceCommandRepository.class, ResourceRevisionRepository.class,
                ResourceCommandService.class,
                AssetCommandSupport.class, AssetFields.class, AssetMutationRepository.class,
                AssetResourcePatchHandler.class,
                AssetResourceCreateHandler.class, AssetResourceDeleteHandler.class, ProjectFields.class,
                ProjectResourcePatchHandler.class,
                StoryboardCommandSupport.class, StoryboardFields.class, StoryboardRelations.class,
                StoryboardVersionView.class,
                StoryboardMutationRepository.class, StoryboardResourcePatchHandler.class,
                ResourceBindMediaHandler.class,
                MediaBindingObserver.class, MediaBindingController.class, MediaContentReader.class,
                ResourceReadRepository.class, ResourceAccessPolicy.class, MediaSourceQueries.class,
                GenerationMediaSourceQueries.class, MediaBusinessReferences.class, MediaPresetReferences.class,
                LegacyMediaRoot.class, MediaRemoteAddressPolicy.class, MediaSourceFetcher.class,
                MediaSourceImportService.class,
                CanvasAccessRepository.class, CanvasAccessPolicy.class, MediaAccessPolicy.class,
                ResourceAgentFence.class, GenerationExecutionContexts.class, GenerationTasks.class,
                GenerationJobRepository.class, MediaJobFence.class, MediaJobRepository.class,
                ComposeWritebackHandler.class,
                MediaBlobLockConfiguration.class, MediaBlobLocks.class, MediaBlobStore.class,
                S3StorageConfigResolver.class,
                StorageProviderRegistry.class, MediaProbe.class, MediaRegistryTransactions.class,
                MediaIntakeService.class,
                MediaMetadataCache.class, MediaResourceQueries.class, MediaAccessTickets.class,
                MediaStorageConfigGuard.class,
                MediaResourceController.class, MediaContentController.class);
        context.refresh();
        intake = context.getBean(MediaIntakeService.class);
        registry = context.getBean(MediaRegistryTransactions.class);
        queries = context.getBean(MediaResourceQueries.class);
        tickets = context.getBean(MediaAccessTickets.class);
        blobs = context.getBean(MediaBlobStore.class);
        http = MockMvcBuilders.standaloneSetup(context.getBean(MediaContentController.class),
                        context.getBean(MediaResourceController.class),
                        context.getBean(MediaBindingController.class), context.getBean(MediaImportController.class),
                        context.getBean(MediaPreviewController.class))
                .setControllerAdvice(new MediaExceptionHandler()).build();
        var image = new BufferedImage(16, 10, BufferedImage.TYPE_INT_RGB);
        image.setRGB(3, 4, 0x0099ff);
        var bytes = new ByteArrayOutputStream();
        ImageIO.write(image, "png", bytes);
        png = bytes.toByteArray();
        context.getBean(S3ClientFactory.class)
                .getClient(context.getBean(S3StorageConfigResolver.class).resolve(config(72)))
                .createBucket(request -> request.bucket("media-test"));
    }

    @BeforeEach
    void setup() {
        jdbc.update("UPDATE afv_storage_config SET is_default = (id = 71), status = 1, deleted = FALSE");
        jdbc.update("UPDATE sys_user SET status = 1 WHERE id IN (71, 72)");
        jdbc.update("DELETE FROM afv_project_member WHERE project_id = 71");
        when(actors.current()).thenReturn(OWNER);
    }

    @AfterEach
    void clearSpy() {
        reset(repository);
        reset(context.getBean(MediaCryptor.class), context.getBean(S3ClientFactory.class));
    }

    @AfterAll
    static void stop() throws Exception {
        if (context != null) {
            context.close();
        }
        if (directory != null) {
            try (var paths = Files.walk(directory)) {
                for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
                    Files.delete(path);
                }
            }
        }
    }

    static MediaResource upload(String request, byte[] bytes, String name, Long project,
                                ResourceCommandContext actor) throws Exception {
        return intake.upload(new ByteArrayInputStream(bytes), name, "image/png", request, project, actor);
    }

    static String id() {
        return UUID.randomUUID().toString();
    }

    static String token(MediaAccessTickets.Issued issued) {
        return issued.url().substring(issued.url().indexOf("?ticket=") + 8);
    }

    static StorageConfig config(long id) {
        return jdbc.query("SELECT * FROM afv_storage_config WHERE id = ? AND deleted = FALSE", (row, index) ->
                        StorageConfig.builder().id(row.getLong("id")).type(row.getString("type"))
                                .provider(row.getString("provider"))
                                .basePath(row.getString("base_path")).endpoint(row.getString("endpoint"))
                                .bucketName(row.getString("bucket_name"))
                                .accessKey(row.getString("access_key")).secretKey(row.getString("secret_key"))
                                .region(row.getString("region"))
                                .isDefault(row.getBoolean("is_default")).status(row.getInt("status")).build(), id).stream()
                .findFirst().orElse(null);
    }
}
