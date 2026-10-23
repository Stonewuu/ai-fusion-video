package com.stonewu.fusion.integration;

import java.sql.Timestamp;
import java.time.Instant;
import com.stonewu.fusion.mapper.resource.ResourceReadMapper;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.mapper.script.ScriptMapper;
import com.stonewu.fusion.mapper.script.ScriptEpisodeMapper;
import com.stonewu.fusion.mapper.script.ScriptSceneItemMapper;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.mapper.resource.ResourceCommandMapper;
import com.stonewu.fusion.mapper.resource.ResourceRevisionMapper;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.config.CacheConfig;
import com.stonewu.fusion.repository.canvas.CanvasAccessRepository;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.repository.resource.ResourceCommandRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.repository.resource.ResourceRevisionRepository;
import com.stonewu.fusion.repository.resource.ProjectWorkspaceRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.CanvasRequestValidator;
import com.stonewu.fusion.service.canvas.CanvasSnapshotCache;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ProjectResourcePatchHandler;
import com.stonewu.fusion.service.resource.ProjectFields;
import com.stonewu.fusion.service.resource.ProjectCreateHandler;
import com.stonewu.fusion.service.resource.ProjectDeleteHandler;
import com.stonewu.fusion.service.resource.ProjectInitializeHandler;
import com.stonewu.fusion.service.resource.ResourceSnapshotCache;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourceDeleteCommand;
import com.stonewu.fusion.service.resource.model.InitializeProjectWorkspaceCommand;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceEventProjectionService;
import com.stonewu.fusion.service.resource.ResourceChangeFeed;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import jakarta.validation.ValidatorFactory;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import com.stonewu.fusion.repository.resource.ScriptMutationRepository;
import com.stonewu.fusion.service.resource.ScriptFields;
import com.stonewu.fusion.service.resource.ScriptCommandSupport;
import com.stonewu.fusion.service.resource.ScriptResourcePatchHandler;
import com.stonewu.fusion.service.resource.ScriptResourceCreateHandler;
import com.stonewu.fusion.service.resource.ScriptResourceDeleteHandler;
import com.stonewu.fusion.service.resource.ScriptSourceHandler;
import com.stonewu.fusion.service.resource.ScriptScenesHandler;
import com.stonewu.fusion.service.resource.model.ScriptSourceCommand;
import com.stonewu.fusion.service.resource.model.ScriptScenesCommand;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.tool.script.ScriptToolCommands;
import com.stonewu.fusion.service.ai.tool.script.ScriptToolQueries;
import com.stonewu.fusion.repository.resource.AssetMutationRepository;
import com.stonewu.fusion.service.resource.AssetFields;
import com.stonewu.fusion.service.resource.AssetCommandSupport;
import com.stonewu.fusion.service.resource.AssetResourceCreateHandler;
import com.stonewu.fusion.service.resource.AssetResourcePatchHandler;
import com.stonewu.fusion.service.resource.AssetResourceDeleteHandler;
import com.stonewu.fusion.repository.resource.AssetReadRepository;
import com.stonewu.fusion.service.resource.AssetResourceQueryService;
import com.stonewu.fusion.service.resource.AssetBatchCreateHandler;
import com.stonewu.fusion.service.resource.model.AssetBatchCreateCommand;
import com.stonewu.fusion.service.ai.tool.asset.AssetToolCommands;
import com.stonewu.fusion.repository.resource.StoryboardMutationRepository;
import com.stonewu.fusion.repository.resource.StoryboardReadRepository;
import com.stonewu.fusion.service.resource.StoryboardFields;
import com.stonewu.fusion.service.resource.StoryboardCommandSupport;
import com.stonewu.fusion.service.resource.StoryboardRelations;
import com.stonewu.fusion.service.resource.StoryboardCreation;
import com.stonewu.fusion.service.resource.StoryboardVersionView;
import com.stonewu.fusion.service.resource.StoryboardResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardResourceCreateHandler;
import com.stonewu.fusion.service.resource.StoryboardResourcePatchHandler;
import com.stonewu.fusion.service.resource.StoryboardResourceDeleteHandler;
import com.stonewu.fusion.service.resource.StoryboardChildrenHandler;
import com.stonewu.fusion.service.resource.StoryboardBatchPatchHandler;
import com.stonewu.fusion.service.resource.model.StoryboardChildrenCommand;
import com.stonewu.fusion.service.resource.model.StoryboardBatchPatchCommand;
import com.stonewu.fusion.service.ai.tool.storyboard.StoryboardToolCommands;
import com.stonewu.fusion.service.ai.tool.storyboard.StoryboardToolQueries;
import com.stonewu.fusion.service.system.SystemConfigService;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.mock;

@Testcontainers(disabledWithoutDocker = false)
class ResourceCommandIT {
    @Container
    static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("resource_commands_test").withUsername("afv").withPassword("afv-test")
            .withUrlParam("connectionTimeZone", "UTC");
    @Container
    static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final CanvasActor ACTOR = new CanvasActor(51, null);
    static final ResourceCommandContext USER = ResourceCommandContext.user(ACTOR, null);
    static AnnotationConfigApplicationContext context;
    static ValidatorFactory validation;
    static JdbcTemplate jdbc;
    static ObjectMapper mapper;
    static ResourceCommandRepository journal;
    static ResourceCommandService commands;
    static ResourceEventProjectionService projection;
    static CanvasQueryService canvases;
    static ResourceAgentFence fence;
    static ResourceReadRepository resources;
    static ResourceQueryService queries;
    static long legacyProject;
    static List<Map<String, Object>> oldHistory;
    static Flyway flyway;

    @BeforeAll
    static void start() throws Exception {
        var source = new DriverManagerDataSource(MYSQL.getJdbcUrl(), MYSQL.getUsername(), MYSQL.getPassword());
        jdbc = new JdbcTemplate(source);
        Flyway.configure().dataSource(source).target("1.1.2.0.0").load().migrate();
        oldHistory = jdbc.queryForList("SELECT version, checksum FROM flyway_schema_history ORDER BY installed_rank");
        jdbc.update("INSERT INTO afv_project(id, name, owner_type, owner_id) VALUES (11, '迁移前项目', 1, 51)");
        legacyProject = 11;
        flyway = Flyway.configure().dataSource(source).load();
        flyway.migrate();
        flyway.validate();
        jdbc.update("INSERT INTO sys_user(id, username, password) VALUES (51, 'resource-owner', 'test-only'), (52, 'resource-other', 'test-only')");
        mapper = new ObjectMapper().findAndRegisterModules();
        CanvasJson json = new CanvasJson(mapper);
        context = new AnnotationConfigApplicationContext();
        var session = MybatisTestSupport.register(context, source);
        fence = spy(new ResourceAgentFence(session.getMapper(AgentRunMapper.class)));
        resources = spy(new ResourceReadRepository(session.getMapper(ResourceReadMapper.class), session.getMapper(ProjectMapper.class),
                session.getMapper(ScriptMapper.class), session.getMapper(ScriptEpisodeMapper.class),
                session.getMapper(ScriptSceneItemMapper.class), json));
        journal = spy(new ResourceCommandRepository(session.getMapper(ResourceCommandMapper.class), json,
                new ResourceRevisionRepository(session.getMapper(ResourceRevisionMapper.class), json, resources)));
        validation = Validation.buildDefaultValidatorFactory();
        context.registerBean(JdbcTemplate.class, () -> jdbc);
        context.registerBean(PlatformTransactionManager.class, () -> new DataSourceTransactionManager(source));
        context.registerBean(ObjectMapper.class, () -> mapper);
        context.registerBean(Validator.class, validation::getValidator);
        context.registerBean(ResourceCommandRepository.class, () -> journal);
        context.registerBean(ResourceAgentFence.class, () -> fence);
        context.registerBean(ResourceReadRepository.class, () -> resources);
        context.registerBean(RedisConnectionFactory.class, () -> new LettuceConnectionFactory(REDIS.getHost(), REDIS.getMappedPort(6379)));
        context.register(CacheConfig.class, CanvasJson.class, CanvasDocumentRepository.class, CanvasAccessRepository.class,
                CanvasAccessPolicy.class, CanvasGraphRepository.class, CanvasJournalRepository.class, CanvasSnapshotCache.class,
                CanvasRequestValidator.class, CanvasQueryService.class, ProjectResourcePatchHandler.class,
                ResourceCommandService.class, ResourceEventProjectionService.class, ResourceChangeFeed.class, ProjectFields.class,
                ProjectWorkspaceRepository.class, ProjectCreateHandler.class, ProjectDeleteHandler.class, ProjectInitializeHandler.class,
                ResourceQueryService.class, ResourceAccessPolicy.class, ResourceSnapshotCache.class, ScriptMutationRepository.class, ScriptFields.class, ScriptCommandSupport.class,
                ScriptResourcePatchHandler.class, ScriptResourceCreateHandler.class, ScriptResourceDeleteHandler.class,
                ScriptSourceHandler.class, ScriptScenesHandler.class, AssetFields.class, AssetCommandSupport.class,
                AssetMutationRepository.class, AssetResourceCreateHandler.class, AssetResourcePatchHandler.class, AssetResourceDeleteHandler.class,
                AssetReadRepository.class, AssetResourceQueryService.class, AssetBatchCreateHandler.class,
                StoryboardMutationRepository.class, StoryboardReadRepository.class, StoryboardFields.class, StoryboardCommandSupport.class, StoryboardRelations.class, StoryboardCreation.class, StoryboardVersionView.class, StoryboardResourceQueryService.class, StoryboardResourceCreateHandler.class, StoryboardResourcePatchHandler.class, StoryboardResourceDeleteHandler.class, StoryboardChildrenHandler.class, StoryboardBatchPatchHandler.class);
        context.refresh();
        commands = context.getBean(ResourceCommandService.class);
        projection = context.getBean(ResourceEventProjectionService.class);
        canvases = context.getBean(CanvasQueryService.class);
        queries = context.getBean(ResourceQueryService.class);
    }

    @AfterEach
    void clearFaults() { reset(journal, fence, resources); }

    @AfterAll
    static void close() {
        if (context != null) context.close();
        if (validation != null) validation.close();
    }

    @Test
    void migrationPreservesHistoryAndInitializesSevenResourceVersions() {
        flyway.validate();
        assertThat(jdbc.queryForObject("SELECT name FROM afv_project WHERE id = ?", String.class, legacyProject)).isEqualTo("迁移前项目");
        assertThat(version("afv_project", legacyProject)).isZero();
        var history = jdbc.queryForList("SELECT version, checksum FROM flyway_schema_history ORDER BY installed_rank");
        assertThat(history.subList(0, oldHistory.size())).containsExactlyElementsOf(oldHistory);
        assertThat(jdbc.queryForObject("""
                SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = DATABASE() AND column_name = 'version'
                AND table_name IN ('afv_project', 'afv_asset', 'afv_asset_item', 'afv_storyboard', 'afv_storyboard_episode', 'afv_storyboard_scene', 'afv_storyboard_item')
                """, Integer.class)).isEqualTo(7);
    }

    @Test
    void atomicallyUpdatesProjectAndDerivedTitlesWithVersionsAuditAndOutbox() {
        long project = project();
        jdbc.update("INSERT INTO afv_script(project_id, title, version) VALUES (?, '旧标题', 0)", project);
        jdbc.update("INSERT INTO afv_storyboard(project_id, title, version) VALUES (?, '旧标题', 0)", project);
        ObjectNode patch = mapper.createObjectNode().put("name", "新项目名").put("coverUrl", "https://example.com/project-cover.png")
                .put("artStyleImageUrl", "/media/images/style.png");
        patch.set("properties", mapper.createObjectNode().put("aspectRatio", "16:9"));
        ResourceCommandReceipt result = commands.patch(command(project, 0, patch), USER);
        assertThat(result.result().path("version").intValue()).isEqualTo(1);
        assertThat(result.result().path("coverUrl").textValue()).isEqualTo("https://example.com/project-cover.png");
        assertThat(result.result().path("artStyleImageUrl").textValue()).isEqualTo("/media/images/style.png");
        assertThat(result.changes()).extracting(change -> change.resourceType()).containsExactly(CanvasValueType.PROJECT, CanvasValueType.SCRIPT, CanvasValueType.STORYBOARD);
        assertThat(jdbc.queryForObject("SELECT title FROM afv_script WHERE project_id = ?", String.class, project)).isEqualTo("新项目名");
        assertThat(jdbc.queryForObject("SELECT title FROM afv_storyboard WHERE project_id = ?", String.class, project)).isEqualTo("新项目名");
        assertThat(jdbc.queryForObject("SELECT version FROM afv_script WHERE project_id = ?", Integer.class, project)).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_event WHERE command_id = ?", Integer.class, result.commandId())).isEqualTo(3);
        assertThat(jdbc.queryForObject("SELECT undo_json FROM afv_resource_command WHERE id = ?", String.class, result.commandId())).contains("旧标题", "project");
    }

    @Test
    void sameRequestReplaysAndDifferentContentOrStaleVersionIsRejected() throws Exception {
        long project = project();
        ResourcePatchCommand command = command(project, 0, mapper.createObjectNode().put("description", "版本一"));
        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
            var first = pool.submit(() -> commands.patch(command, USER));
            var second = pool.submit(() -> commands.patch(command, USER));
            assertThat(first.get(15, TimeUnit.SECONDS)).isEqualTo(second.get(15, TimeUnit.SECONDS));
        }
        assertThat(version("afv_project", project)).isEqualTo(1);
        rejects(() -> commands.patch(new ResourcePatchCommand(command.requestId(), CanvasValueType.PROJECT, project, 0,
                mapper.createObjectNode().put("description", "另外内容")), USER), CanvasErrorCode.REQUEST_CONFLICT);
        rejects(() -> commands.patch(command(project, 0, mapper.createObjectNode().put("description", "过期")), USER), CanvasErrorCode.VERSION_CONFLICT);
    }

    @Test
    void explicitClearingWorksAndInvalidOrPrivilegedFieldsCannotBeWritten() {
        long project = project();
        commands.patch(command(project, 0, mapper.createObjectNode().put("artStyle", "custom")), USER);
        commands.patch(command(project, 1, mapper.createObjectNode().putNull("artStyle")), USER);
        assertThat(jdbc.queryForObject("SELECT art_style FROM afv_project WHERE id = ?", String.class, project)).isNull();
        for (ObjectNode invalid : List.of(mapper.createObjectNode().put("ownerId", 52), mapper.createObjectNode().put("name", " "),
                mapper.createObjectNode().put("coverUrl", "data:image/png;base64,AAAA"), mapper.createObjectNode().put("status", 1.5))) {
            rejects(() -> commands.patch(command(project, 2, invalid), USER), CanvasErrorCode.INVALID_COMMAND);
        }
        assertThat(version("afv_project", project)).isEqualTo(2);
    }

    @Test
    void commandJournalFailureRollsBackContentVersionsAndAdmission() {
        long project = project();
        ResourcePatchCommand command = command(project, 0, mapper.createObjectNode().put("name", "不能提交"));
        doThrow(new IllegalStateException("模拟审计失败")).when(journal).commit(any(), any());
        assertThatThrownBy(() -> commands.patch(command, USER)).hasMessage("模拟审计失败");
        assertThat(version("afv_project", project)).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE request_id = ?", Integer.class, command.requestId())).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_event WHERE project_id = ?", Integer.class, project)).isZero();
    }

    @Test
    void projectedEventsAreDurableExactlyOnceAndDoNotReachAnotherProject() {
        drain();
        long project = project();
        String first = canvas(project);
        String second = canvas(project);
        String unrelated = canvas(project());
        var old = canvases.snapshot(first, ACTOR);
        var receipt = commands.patch(command(project, 0, mapper.createObjectNode().put("description", "共享修改")), USER);
        assertThat(projection.publishNext()).isTrue();
        assertThat(projection.publishNext()).isFalse();
        var events = canvases.events(first, ACTOR, old.lastSequence(), 10).events();
        assertThat(events).hasSize(1);
        assertThat(events.getFirst().type()).isEqualTo("resource.changed");
        assertThat(events.getFirst().payload().path("resourceCommandId").textValue()).isEqualTo(receipt.commandId());
        assertThat(canvases.events(second, ACTOR, 0, 10).events()).hasSize(1);
        assertThat(canvases.events(unrelated, ACTOR, 0, 10).events()).isEmpty();
    }

    @Test
    void failedProjectionRollsBackEveryCanvasAndCanResumeWithoutDuplicates() {
        drain();
        long project = project();
        String first = canvas(project);
        String second = canvas(project);
        commands.patch(command(project, 0, mapper.createObjectNode().put("name", "重试投递")), USER);
        doThrow(new IllegalStateException("模拟投递提交失败")).when(journal).markPublished(anyLong());
        assertThatThrownBy(projection::publishNext).hasMessage("模拟投递提交失败");
        assertThat(canvases.snapshot(first, ACTOR).lastSequence()).isZero();
        assertThat(canvases.snapshot(second, ACTOR).lastSequence()).isZero();
        reset(journal);
        assertThat(projection.publishNext()).isTrue();
        assertThat(canvases.events(first, ACTOR, 0, 10).events()).hasSize(1);
        assertThat(canvases.events(second, ACTOR, 0, 10).events()).hasSize(1);
    }

    @Test
    void permissionAndCanvasScopeAreCheckedBeforeIdempotentReplay() {
        long project = project();
        ResourcePatchCommand command = command(project, 0, mapper.createObjectNode().put("name", "授权修改"));
        commands.patch(command, USER);
        var other = ResourceCommandContext.user(new CanvasActor(52, null), null);
        rejects(() -> commands.patch(command, other), CanvasErrorCode.ACCESS_DENIED);
        var wrongCanvas = ResourceCommandContext.user(ACTOR, canvas(project()));
        rejects(() -> commands.patch(command(project, 1, mapper.createObjectNode().put("name", "跨画布")), wrongCanvas), CanvasErrorCode.ACCESS_DENIED);
        jdbc.update("UPDATE afv_project SET owner_id = 52 WHERE id = ?", project);
        rejects(() -> commands.patch(command, USER), CanvasErrorCode.ACCESS_DENIED);
    }

    @Test
    void sameTransactionPermissionChecksObserveCommittedRevocation() {
        long project = project();
        var member = new CanvasActor(52, null);
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (?, ?, 3)",
                project, member.userId());
        var manager = context.getBean(PlatformTransactionManager.class);
        var reads = new TransactionTemplate(manager);
        reads.setIsolationLevel(TransactionDefinition.ISOLATION_READ_COMMITTED);
        var revocation = new TransactionTemplate(manager);
        revocation.setPropagationBehavior(TransactionDefinition.PROPAGATION_REQUIRES_NEW);
        var access = context.getBean(CanvasAccessPolicy.class);
        reads.executeWithoutResult(ignored -> {
            access.requireProjectEditor(project, member);
            revocation.executeWithoutResult(inner -> jdbc.update(
                    "UPDATE afv_project_member SET role = 4 WHERE project_id = ? AND user_id = ?",
                    project, member.userId()));
            rejects(() -> access.requireProjectEditor(project, member), CanvasErrorCode.ACCESS_DENIED);
        });
    }

    @Test
    void agentWritesAreFencedAndCanReplayAcrossAValidOwnerEpochChange() {
        long project = project();
        String run = agentRun(project);
        var owner = new ResourceCommandContext(ACTOR, "agent", null, run, "call-1", "owner", 1L);
        var oldOwner = new ResourceCommandContext(ACTOR, "agent", null, run, "call-1", "owner", 2L);
        ResourcePatchCommand command = command(project, 0, mapper.createObjectNode().put("description", "智能体修改"));
        rejects(() -> commands.patch(command, oldOwner), CanvasErrorCode.ACCESS_DENIED);
        var receipt = commands.patch(command, owner);
        jdbc.update("UPDATE afv_agent_run SET owner_epoch = 2 WHERE run_id = ?", run);
        rejects(() -> commands.patch(command, owner), CanvasErrorCode.ACCESS_DENIED);
        assertThat(commands.patch(command, oldOwner)).isEqualTo(receipt);
        jdbc.update("UPDATE afv_agent_run SET status = 'CANCEL_REQUESTED', cancel_requested_at = UTC_TIMESTAMP(3) WHERE run_id = ?", run);
        rejects(() -> commands.patch(command, oldOwner), CanvasErrorCode.ACCESS_DENIED);
        assertThat(version("afv_project", project)).isEqualTo(1);
    }

    @Test
    void teamAgentKeepsItsPersistedScopeAndLosesAccessWhenMembershipIsRevoked() {
        long project = project();
        jdbc.update("INSERT INTO afv_team(id, name, owner_user_id) VALUES (801, '运行所属团队', 51), (802, '另一个团队', 51)");
        jdbc.update("INSERT INTO afv_team_member(team_id, user_id, role) VALUES (801, 51, 1), (802, 51, 1)");
        jdbc.update("UPDATE afv_project SET owner_type = 2, owner_id = 801 WHERE id = ?", project);
        String run = agentRun(project);
        jdbc.update("UPDATE afv_agent_run SET actor_team_id = 801 WHERE run_id = ?", run);
        var team = new ResourceCommandContext(new CanvasActor(51, 801L), "agent", null, run, "call-team", "owner", 1L);
        var otherTeam = new ResourceCommandContext(new CanvasActor(51, 802L), "agent", null, run, "call-team", "owner", 1L);
        var command = command(project, 0, mapper.createObjectNode().put("name", "团队修改"));
        commands.patch(command, team);
        rejects(() -> commands.patch(command, otherTeam), CanvasErrorCode.ACCESS_DENIED);
        jdbc.update("UPDATE afv_team_member SET status = 0 WHERE team_id = 801 AND user_id = 51");
        rejects(() -> commands.patch(command, team), CanvasErrorCode.ACCESS_DENIED);
    }

    @Test
    void aLateAgentLeaseExpiryRollsBackTheBusinessChange() {
        long project = project();
        String run = agentRun(project);
        var agent = new ResourceCommandContext(ACTOR, "agent", null, run, "call-late", "owner", 1L);
        AtomicInteger checks = new AtomicInteger();
        doAnswer(invocation -> {
            if (checks.incrementAndGet() == 2) {
                assertThat(jdbc.queryForObject("SELECT name FROM afv_project WHERE id = ?", String.class, project)).isEqualTo("不得落库");
                jdbc.update("UPDATE afv_agent_run SET lease_until = DATE_SUB(UTC_TIMESTAMP(3), INTERVAL 1 SECOND) WHERE run_id = ?", run);
            }
            return invocation.callRealMethod();
        }).when(fence).requireActive(any(), anyLong());
        rejects(() -> commands.patch(command(project, 0, mapper.createObjectNode().put("name", "不得落库")), agent), CanvasErrorCode.ACCESS_DENIED);
        assertThat(version("afv_project", project)).isZero();
    }

    @Test
    void createAtomicallyBuildsOwnedWorkspaceAndReplaysWithoutDuplicates() throws Exception {
        ObjectNode fields = mapper.createObjectNode().put("name", "一次创建的项目").put("artStyle", "custom")
                .put("coverUrl", "https://example.com/cover.png");
        fields.set("properties", mapper.createObjectNode().put("aspectRatio", "16:9"));
        var command = new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, null, fields);
        ResourceCommandReceipt result;
        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
            var first = pool.submit(() -> commands.execute(command, USER));
            var second = pool.submit(() -> commands.execute(command, USER));
            result = first.get(15, TimeUnit.SECONDS);
            assertThat(second.get(15, TimeUnit.SECONDS)).isEqualTo(result);
        }
        long id = result.result().path("id").longValue();
        assertThat(result.result().path("ownerId").longValue()).isEqualTo(ACTOR.userId());
        assertThat(result.result().path("properties").path("aspectRatio").textValue()).isEqualTo("16:9");
        assertThat(result.changes()).hasSize(3);
        assertThat(jdbc.queryForObject("""
                SELECT ABS(TIMESTAMPDIFF(SECOND, p.create_time, m.create_time))
                FROM afv_project p JOIN afv_project_member m ON m.project_id = p.id
                WHERE p.id = ? AND m.user_id = ? AND m.role = 1 AND m.deleted = FALSE
                """, Long.class, id, ACTOR.userId())).isLessThan(10);
        ObjectNode workspace = queries.workspace(id, ACTOR);
        assertThat(workspace.path("script").path("title").textValue()).isEqualTo("一次创建的项目");
        assertThat(workspace.path("storyboard").path("scriptId").longValue()).isEqualTo(workspace.path("script").path("id").longValue());
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_script WHERE project_id = ?", Integer.class, id)).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_storyboard WHERE project_id = ?", Integer.class, id)).isEqualTo(1);
        assertThat(queries.get(CanvasValueType.PROJECT, id, ACTOR).header().version()).isZero();
    }

    @Test
    void failedCreationRollsBackProjectContainersMembershipAndEvents() {
        String name = "回滚创建-" + UUID.randomUUID();
        var request = new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, null, mapper.createObjectNode().put("name", name));
        long scriptsBefore = jdbc.queryForObject("SELECT COUNT(*) FROM afv_script", Long.class);
        long membersBefore = jdbc.queryForObject("SELECT COUNT(*) FROM afv_project_member", Long.class);
        doThrow(new IllegalStateException("创建提交失败")).when(journal).commit(any(), any());
        assertThatThrownBy(() -> commands.execute(request, USER)).hasMessage("创建提交失败");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_project WHERE name = ?", Integer.class, name)).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_script", Long.class)).isEqualTo(scriptsBefore);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_project_member", Long.class)).isEqualTo(membersBefore);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE request_id = ?", Integer.class, request.requestId())).isZero();
    }

    @Test
    void createRejectsInlineMediaAndOwnershipInjection() {
        for (ObjectNode invalid : List.of(mapper.createObjectNode().put("name", "错误图片").put("coverUrl", "data:image/png;base64,AA"),
                mapper.createObjectNode().put("name", "伪造归属").put("ownerId", 52))) {
            rejects(() -> commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, null, invalid), USER),
                    CanvasErrorCode.INVALID_COMMAND);
        }
    }

    @Test
    void initializationOnlyFillsMissingContainersAndPreservesScriptContent() {
        long project = project();
        jdbc.update("INSERT INTO afv_script(project_id, title, raw_content, version) VALUES (?, '现有剧本', '不要覆盖的原文', 4)", project);
        var command = new InitializeProjectWorkspaceCommand(UUID.randomUUID().toString(), project, 0);
        var result = commands.execute(command, USER);
        assertThat(commands.execute(command, USER)).isEqualTo(result);
        assertThat(result.result().path("script").path("rawContent").textValue()).isEqualTo("不要覆盖的原文");
        assertThat(result.result().path("script").path("version").intValue()).isEqualTo(4);
        assertThat(result.result().path("projectVersion").intValue()).isEqualTo(1);
        assertThat(result.changes()).extracting(change -> change.resourceType()).containsExactly(CanvasValueType.STORYBOARD, CanvasValueType.PROJECT);
        var repeat = commands.execute(new InitializeProjectWorkspaceCommand(UUID.randomUUID().toString(), project, 1), USER);
        assertThat(repeat.changes()).isEmpty();
        assertThat(repeat.result()).isEqualTo(result.result());
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_script WHERE project_id = ?", Integer.class, project)).isEqualTo(1);
    }

    @Test
    void deletionVersionsAndAuditsEveryBusinessTypeAndLeavesAnotherProjectAlone() {
        long project = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, null,
                mapper.createObjectNode().put("name", "完整删除范围")), USER).result().path("id").longValue();
        ObjectNode workspace = queries.workspace(project, ACTOR);
        long script = workspace.path("script").path("id").longValue();
        long board = workspace.path("storyboard").path("id").longValue();
        jdbc.update("INSERT INTO afv_script_episode(script_id, episode_number, title, version) VALUES (?, 1, '第一集', 0)", script);
        long episode = jdbc.queryForObject("SELECT id FROM afv_script_episode WHERE script_id = ?", Long.class, script);
        jdbc.update("INSERT INTO afv_script_scene_item(script_id, episode_id, version) VALUES (?, ?, 0)", script, episode);
        jdbc.update("INSERT INTO afv_storyboard_episode(storyboard_id, episode_number) VALUES (?, 1)", board);
        long boardEpisode = jdbc.queryForObject("SELECT id FROM afv_storyboard_episode WHERE storyboard_id = ?", Long.class, board);
        jdbc.update("INSERT INTO afv_storyboard_scene(storyboard_id, episode_id) VALUES (?, ?)", board, boardEpisode);
        long scene = jdbc.queryForObject("SELECT id FROM afv_storyboard_scene WHERE storyboard_id = ?", Long.class, board);
        jdbc.update("INSERT INTO afv_storyboard_item(storyboard_id, storyboard_episode_id, storyboard_scene_id) VALUES (?, ?, ?)", board, boardEpisode, scene);
        jdbc.update("INSERT INTO afv_asset(project_id, name, type) VALUES (?, '角色', 'character')", project);
        long asset = jdbc.queryForObject("SELECT id FROM afv_asset WHERE project_id = ?", Long.class, project);
        jdbc.update("INSERT INTO afv_asset_item(asset_id) VALUES (?)", asset);
        long unrelated = project();
        queries.get(CanvasValueType.PROJECT, project, ACTOR);
        var command = new ResourceDeleteCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, project, 0);
        var result = commands.execute(command, USER);
        assertThat(commands.execute(command, USER)).isEqualTo(result);
        assertThat(result.changes()).hasSize(10).allMatch(change -> change.deleted() && change.version() == 1);
        assertThat(result.result().path("affectedCount").intValue()).isEqualTo(10);
        assertThat(jdbc.queryForObject("""
                SELECT ABS(TIMESTAMPDIFF(SECOND, update_time, CURRENT_TIMESTAMP))
                FROM afv_project_member WHERE project_id = ? AND user_id = ? AND deleted = TRUE
                """, Long.class, project, ACTOR.userId())).isLessThan(10);
        assertThat(jdbc.queryForObject("SELECT undo_json FROM afv_resource_command WHERE id = ?", String.class, result.commandId()))
                .contains("members", "完整删除范围", "第一集", "角色");
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_project WHERE id = ?", Boolean.class, unrelated)).isFalse();
        rejects(() -> queries.get(CanvasValueType.PROJECT, project, ACTOR), CanvasErrorCode.RESOURCE_DELETED);
        assertThat(queries.projects(ACTOR)).noneMatch(item -> item.path("id").longValue() == project);
    }

    @Test
    void deleteRequiresManagementAndCurrentVersionAndRollsBackOnJournalFailure() {
        long project = project();
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (?, 52, 3)", project);
        var other = ResourceCommandContext.user(new CanvasActor(52, null), null);
        rejects(() -> commands.execute(new ResourceDeleteCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, project, 0), other), CanvasErrorCode.ACCESS_DENIED);
        rejects(() -> commands.execute(new ResourceDeleteCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, project, 1), USER), CanvasErrorCode.VERSION_CONFLICT);
        doThrow(new IllegalStateException("删除提交失败")).when(journal).commit(any(), any());
        assertThatThrownBy(() -> commands.execute(new ResourceDeleteCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, project, 0), USER))
                .hasMessage("删除提交失败");
        assertThat(version("afv_project", project)).isZero();
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_project WHERE id = ?", Boolean.class, project)).isFalse();
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_project_member WHERE project_id = ? AND user_id = 52", Boolean.class, project)).isFalse();
    }

    @Test
    void teamMemberCreatingAProjectBecomesItsManager() {
        jdbc.update("INSERT INTO afv_team(id, name, owner_user_id) VALUES (9300, '创作者团队', 52)");
        jdbc.update("INSERT INTO afv_team_member(team_id, user_id, role) VALUES (9300, 51, 3), (9300, 52, 1)");
        var actor = new CanvasActor(51, 9300L);
        var scope = ResourceCommandContext.user(actor, null);
        var result = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, null,
                mapper.createObjectNode().put("name", "成员创建的项目")), scope);
        long project = result.result().path("id").longValue();
        assertThat(result.result().path("ownerId").longValue()).isEqualTo(9300);
        assertThat(queries.get(CanvasValueType.PROJECT, project, actor).data().path("ownerType").intValue()).isEqualTo(2);
        assertThat(commands.execute(new ResourceDeleteCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, project, 0), scope)
                .result().path("deleted").booleanValue()).isTrue();
    }

    @Test
    void resourceSnapshotCacheCannotResurrectAnEarlierVersionAndChecksAuthorization() throws Exception {
        long project = project();
        CountDownLatch loaded = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicBoolean once = new AtomicBoolean(true);
        doAnswer(invocation -> {
            Object result = invocation.callRealMethod();
            ResourceHeader header = invocation.getArgument(0);
            if (header.resourceType() == CanvasValueType.PROJECT && header.resourceId() == project && once.compareAndSet(true, false)) {
                loaded.countDown();
                assertThat(release.await(10, TimeUnit.SECONDS)).isTrue();
            }
            return result;
        }).when(resources).data(any());
        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
            var old = pool.submit(() -> queries.get(CanvasValueType.PROJECT, project, ACTOR));
            try {
                assertThat(loaded.await(10, TimeUnit.SECONDS)).isTrue();
                commands.patch(command(project, 0, mapper.createObjectNode().put("name", "读取新版本")), USER);
            } finally { release.countDown(); }
            assertThat(old.get(10, TimeUnit.SECONDS).header().version()).isZero();
        }
        var latest = queries.get(CanvasValueType.PROJECT, project, ACTOR);
        assertThat(latest.header().version()).isEqualTo(1);
        assertThat(latest.data().path("name").textValue()).isEqualTo("读取新版本");
        rejects(() -> queries.get(CanvasValueType.PROJECT, project, new CanvasActor(52, null)), CanvasErrorCode.ACCESS_DENIED);
        assertThat(queries.projectPage(new CanvasActor(52, null), 1, 100).getList()).noneMatch(item -> item.path("id").longValue() == project);
    }


    @Test
    void scriptTreeWritesUseObservedVersionsAndEmitParentChanges() {
        long script = script();
        var create = new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_EPISODE, null,
                mapper.createObjectNode().put("scriptId", script).put("parentVersion", 0).put("episodeNumber", 1).put("title", "初稿"));
        var created = commands.execute(create, USER);
        assertThat(commands.execute(create, USER)).isEqualTo(created);
        long episode = created.result().path("id").longValue();
        long scene = scene(episode, 0, "客厅");
        assertThat(version("afv_script", script)).isEqualTo(2);
        assertThat(version("afv_script_episode", episode)).isEqualTo(1);
        assertThat(queries.get(CanvasValueType.SCRIPT, script, ACTOR).data().path("totalEpisodes").intValue()).isEqualTo(1);
        var changed = commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_SCENE, scene, 0,
                mapper.createObjectNode().put("location", "雨夜").putNull("sceneDescription")), USER);
        assertThat(changed.changes()).extracting(c -> c.resourceType()).containsExactly(CanvasValueType.SCRIPT_SCENE, CanvasValueType.SCRIPT_EPISODE, CanvasValueType.SCRIPT);
        assertThat(changed.result().path("location").textValue()).isEqualTo("雨夜");
        assertThat(changed.result().get("sceneDescription").isNull()).isTrue();
        rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_SCENE, scene, 0,
                mapper.createObjectNode().put("location", "过期内容")), USER), CanvasErrorCode.VERSION_CONFLICT);
        rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT, script, 3,
                mapper.createObjectNode().put("title", "不能改固定标题")), USER), CanvasErrorCode.INVALID_COMMAND);
        rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_EPISODE, episode, 2,
                mapper.createObjectNode().put("scriptId", 99)), USER), CanvasErrorCode.INVALID_COMMAND);
        assertThat(queries.scriptTree(CanvasValueType.SCRIPT_EPISODE, episode, ACTOR).path("scenes").get(0).path("version").intValue()).isEqualTo(1);
    }

    @Test
    void scriptEpisodeDeletionIsAtomicCascadingAuditedAndReplayable() {
        long script = script();
        long episode = episode(script, 0, 1);
        long scene = scene(episode, 0, "需要保留在审计中的对白");
        var deletion = new ResourceDeleteCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_EPISODE, episode, 1);
        doThrow(new IllegalStateException("级联提交失败")).when(journal).commit(any(), any());
        assertThatThrownBy(() -> commands.execute(deletion, USER)).hasMessage("级联提交失败");
        assertThat(queries.get(CanvasValueType.SCRIPT_SCENE, scene, ACTOR).header().version()).isZero();
        reset(journal);
        var receipt = commands.execute(deletion, USER);
        assertThat(commands.execute(deletion, USER)).isEqualTo(receipt);
        assertThat(receipt.changes()).filteredOn(c -> c.deleted()).extracting(c -> c.resourceType())
                .containsExactly(CanvasValueType.SCRIPT_SCENE, CanvasValueType.SCRIPT_EPISODE);
        rejects(() -> queries.get(CanvasValueType.SCRIPT_SCENE, scene, ACTOR), CanvasErrorCode.RESOURCE_DELETED);
        assertThat(jdbc.queryForObject("SELECT undo_json FROM afv_resource_command WHERE id = ?", String.class, receipt.commandId()))
                .contains("需要保留在审计中的对白");
        assertThat(queries.get(CanvasValueType.SCRIPT, script, ACTOR).data().path("totalEpisodes").intValue()).isZero();
    }

    @Test
    void sourceReplacementDetectsDescendantEditsAndKeepsHistoricalReferences() {
        long script = script();
        long episode = episode(script, 0, 1);
        long scene = scene(episode, 0, "被引用的旧场次");
        int viewedVersion = version("afv_script", script);
        commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_SCENE, scene, 0,
                mapper.createObjectNode().put("sceneDescription", "另一个用户刚刚修改")), USER);
        rejects(() -> commands.execute(new ScriptSourceCommand(UUID.randomUUID().toString(), script, viewedVersion, "新原文"), USER),
                CanvasErrorCode.VERSION_CONFLICT);
        var replacement = new ScriptSourceCommand(UUID.randomUUID().toString(), script, viewedVersion + 1, "新原文");
        var receipt = commands.execute(replacement, USER);
        assertThat(commands.execute(replacement, USER)).isEqualTo(receipt);
        assertThat(receipt.result().path("rawContent").textValue()).isEqualTo("新原文");
        assertThat(receipt.result().path("totalEpisodes").intValue()).isZero();
        assertThat(receipt.changes()).filteredOn(c -> c.deleted()).hasSize(2);
        assertThat(jdbc.queryForObject("SELECT scene_description FROM afv_script_scene_item WHERE id = ?", String.class, scene))
                .isEqualTo("另一个用户刚刚修改");
        assertThat(queries.scriptChildren(CanvasValueType.SCRIPT_EPISODE, script, ACTOR)).isEmpty();
    }

    @Test
    void sceneBatchesRejectPartialOrForeignOrdersAndCommitOnlyOnce() {
        long script = script();
        long episode = episode(script, 0, 1);
        var command = new ScriptScenesCommand(UUID.randomUUID().toString(), episode, 0, ScriptScenesCommand.Mode.APPEND,
                List.of(mapper.createObjectNode().put("sceneHeading", "第一场"), mapper.createObjectNode().put("sceneHeading", "第二场")), List.of());
        var saved = commands.execute(command, USER);
        assertThat(commands.execute(command, USER)).isEqualTo(saved);
        List<ObjectNode> scenes = queries.scriptChildren(CanvasValueType.SCRIPT_SCENE, episode, ACTOR);
        List<Long> ids = scenes.stream().map(scene -> scene.path("id").longValue()).toList();
        assertThat(scenes).hasSize(2);
        rejects(() -> commands.execute(new ScriptScenesCommand(UUID.randomUUID().toString(), episode, 1, ScriptScenesCommand.Mode.REORDER,
                List.of(), List.of(ids.get(0))), USER), CanvasErrorCode.INVALID_COMMAND);
        rejects(() -> commands.execute(new ScriptScenesCommand(UUID.randomUUID().toString(), episode, 1, ScriptScenesCommand.Mode.REORDER,
                List.of(), List.of(ids.get(0), ids.get(0))), USER), CanvasErrorCode.INVALID_COMMAND);
        var sorted = commands.execute(new ScriptScenesCommand(UUID.randomUUID().toString(), episode, 1, ScriptScenesCommand.Mode.REORDER,
                List.of(), List.of(ids.get(1), ids.get(0))), USER);
        assertThat(sorted.result().path("scenes").get(0).path("id").longValue()).isEqualTo(ids.get(1));
        assertThat(sorted.result().path("episode").path("version").intValue()).isEqualTo(2);
        rejects(() -> commands.execute(new ScriptScenesCommand(UUID.randomUUID().toString(), episode, 1, ScriptScenesCommand.Mode.REPLACE,
                List.of(mapper.createObjectNode().put("sceneHeading", "过期批次")), List.of()), USER), CanvasErrorCode.VERSION_CONFLICT);
        rejects(() -> commands.execute(new ScriptScenesCommand(UUID.randomUUID().toString(), episode, 2, ScriptScenesCommand.Mode.APPEND,
                List.of(mapper.createObjectNode().put("sceneHeading", "不能局部新增"), mapper.createObjectNode().put("ownerId", 51)), List.of()), USER), CanvasErrorCode.INVALID_COMMAND);
        assertThat(queries.scriptChildren(CanvasValueType.SCRIPT_SCENE, episode, ACTOR)).hasSize(2);
        assertThat(version("afv_script_episode", episode)).isEqualTo(2);
    }

    @Test
    void differentSceneEditsSerializeTheirParentVersionsWithoutLosingEitherEdit() throws Exception {
        long script = script();
        long episode = episode(script, 0, 1);
        long first = scene(episode, 0, "第一场");
        long second = scene(episode, 1, "第二场");
        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
            var a = pool.submit(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_SCENE, first, 0,
                    mapper.createObjectNode().put("location", "城市")), USER));
            var b = pool.submit(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_SCENE, second, 0,
                    mapper.createObjectNode().put("location", "山林")), USER));
            assertThat(a.get(15, TimeUnit.SECONDS).result().path("location").textValue()).isEqualTo("城市");
            assertThat(b.get(15, TimeUnit.SECONDS).result().path("location").textValue()).isEqualTo("山林");
        }
        assertThat(version("afv_script_episode", episode)).isEqualTo(4);
        assertThat(version("afv_script", script)).isEqualTo(5);
    }

    @Test
    void scriptCommandsRejectCrossProjectAssetsAndUnauthorizedReads() {
        long script = script();
        long project = queries.get(CanvasValueType.SCRIPT, script, ACTOR).header().projectId();
        long episode = episode(script, 0, 1);
        long scene = scene(episode, 0, "测试关联");
        long otherProject = project();
        jdbc.update("INSERT INTO afv_asset(project_id, name, type) VALUES (?, '其他项目的场景', 'scene')", otherProject);
        long asset = jdbc.queryForObject("SELECT id FROM afv_asset WHERE project_id = ?", Long.class, otherProject);
        rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_SCENE, scene, 0,
                mapper.createObjectNode().put("sceneAssetId", asset)), USER), CanvasErrorCode.INVALID_COMMAND);
        rejects(() -> queries.scriptTree(CanvasValueType.SCRIPT, script, new CanvasActor(52, null)), CanvasErrorCode.ACCESS_DENIED);
        rejects(() -> commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_EPISODE, otherProject,
                mapper.createObjectNode().put("scriptId", script).put("parentVersion", 2).put("episodeNumber", 2)), USER), CanvasErrorCode.INVALID_COMMAND);
        assertThat(version("afv_script_scene_item", scene)).isZero();
        assertThat(queries.projectScriptTree(project, true, ACTOR).path("episodes").get(0).path("scenes").size()).isEqualTo(1);
    }


    @Test
    void scriptAgentToolsReadVersionsApplyRealCommandsAndReplayWithTrustedRunIdentity() {
        long script = script();
        long project = queries.get(CanvasValueType.SCRIPT, script, ACTOR).header().projectId();
        String run = agentRun(project);
        var mutation = new ResourceCommandContext(ACTOR, "agent", null, run, "save-episode", "owner", 1L);
        var actor = ToolExecutionContext.forInvocation(51L, 1, 51L, mutation);
        var json = context.getBean(CanvasJson.class);
        var queryTools = new ScriptToolQueries(queries, json);
        var tools = new ScriptToolCommands(commands, queries, json);
        ObjectNode read = json.read(queryTools.execute("get_script", "{\"scriptId\":" + script + "}", actor), ObjectNode.class);
        assertThat(read.path("scriptVersion").intValue()).isZero();
        String input = "{\"scriptId\":" + script + ",\"expectedVersion\":0,\"episodeNumber\":1,\"title\":\"Agent 创建\"}";
        ObjectNode created = json.read(tools.saveEpisode(input, actor), ObjectNode.class);
        assertThat(created.path("status").textValue()).isEqualTo("success");
        assertThat(json.read(tools.saveEpisode(input, actor), ObjectNode.class)).isEqualTo(created);
        long episode = created.path("scriptEpisodeId").longValue();
        var sceneActor = ToolExecutionContext.forInvocation(51L, 1, 51L,
                new ResourceCommandContext(ACTOR, "agent", null, run, "save-scenes", "owner", 1L));
        String sceneInput = "{\"scriptEpisodeId\":" + episode + ",\"episode_version\":0,\"scenes\":[{\"scene_heading\":\"内景 客厅 夜\",\"dialogues\":[{\"type\":1,\"character_name\":\"甲\",\"content\":\"你好\"}]}]}";
        ObjectNode saved = json.read(tools.saveScenes(sceneInput, sceneActor), ObjectNode.class);
        assertThat(saved.path("status").textValue()).isEqualTo("success");
        assertThat(saved.path("episode_version").intValue()).isEqualTo(1);
        ObjectNode detail = json.read(queryTools.execute("get_script_episode",
                "{\"scriptEpisodeId\":" + episode + ",\"detailLevel\":\"full\"}", actor), ObjectNode.class);
        assertThat(detail.path("scenes").get(0).path("dialogues").get(0).path("content").textValue()).isEqualTo("你好");
        assertThat(detail.path("scenes").get(0).has("version")).isTrue();
        assertThat(tools.patch(CanvasValueType.SCRIPT, "scriptId", "{\"scriptId\":" + script + ",\"content\":\"缺少版本\"}", actor)).contains("expectedVersion");
        jdbc.update("UPDATE afv_agent_run SET status = 'CANCEL_REQUESTED', cancel_requested_at = UTC_TIMESTAMP(3) WHERE run_id = ?", run);
        assertThat(tools.saveScenes(sceneInput, sceneActor)).contains("error");
        assertThat(queries.scriptChildren(CanvasValueType.SCRIPT_SCENE, episode, ACTOR)).hasSize(1);
    }

    @Test
    void insertSceneKeepsAllPositionsAtomicAndVersionsMovedSiblings() {
        long script = script();
        long episode = episode(script, 0, 1);
        long first = scene(episode, 0, "第一场");
        long second = scene(episode, 1, "第二场");
        var request = new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_SCENE, null,
                mapper.createObjectNode().put("episodeId", episode).put("parentVersion", 2).put("sortOrder", 0).put("sceneHeading", "插入开场"));
        var receipt = commands.execute(request, USER);
        var scenes = queries.scriptChildren(CanvasValueType.SCRIPT_SCENE, episode, ACTOR);
        assertThat(scenes).extracting(item -> item.path("id").longValue()).containsExactly(receipt.result().path("id").longValue(), first, second);
        assertThat(scenes).extracting(item -> item.path("sortOrder").intValue()).containsExactly(0, 1, 2);
        assertThat(scenes).extracting(item -> item.path("version").intValue()).containsExactly(0, 1, 1);
        assertThat(commands.execute(request, USER)).isEqualTo(receipt);
    }


    @Test
    void globalAssetReadsUseTypedOwnershipAndPersistedTeamScopeAndRejectDeletedParents() {
        jdbc.update("INSERT INTO afv_team(id, name, owner_user_id) VALUES (51, '团队与用户编号相同', 52)");
        jdbc.update("INSERT INTO afv_team_member(team_id, user_id, role) VALUES (51, 52, 1)");
        jdbc.update("INSERT INTO afv_asset(name, type, user_id, owner_type, owner_id) VALUES ('不能把团队ID当用户ID', 'character', 51, 2, 51)");
        long teamAsset = jdbc.queryForObject("SELECT id FROM afv_asset WHERE name = '不能把团队ID当用户ID'", Long.class);
        rejects(() -> queries.get(CanvasValueType.ASSET, teamAsset, ACTOR), CanvasErrorCode.ACCESS_DENIED);
        var teamActor = new CanvasActor(52, 51L);
        assertThat(queries.get(CanvasValueType.ASSET, teamAsset, teamActor).header().projectId()).isNull();
        jdbc.update("INSERT INTO afv_asset_item(asset_id, name) VALUES (?, '团队变体')", teamAsset);
        long item = jdbc.queryForObject("SELECT id FROM afv_asset_item WHERE asset_id = ?", Long.class, teamAsset);
        assertThat(queries.get(CanvasValueType.ASSET_ITEM, item, teamActor).data().path("name").textValue()).isEqualTo("团队变体");
        jdbc.update("UPDATE afv_team_member SET status = 0 WHERE team_id = 51 AND user_id = 52");
        rejects(() -> queries.get(CanvasValueType.ASSET_ITEM, item, teamActor), CanvasErrorCode.ACCESS_DENIED);
        jdbc.update("UPDATE afv_team_member SET status = 1 WHERE team_id = 51 AND user_id = 52");
        jdbc.update("UPDATE afv_asset SET deleted = TRUE WHERE id = ?", teamAsset);
        rejects(() -> queries.get(CanvasValueType.ASSET_ITEM, item, teamActor), CanvasErrorCode.RESOURCE_DELETED);
        jdbc.update("INSERT INTO afv_asset(name, type, user_id, owner_type, owner_id) VALUES ('个人全局资产', 'prop', 51, 1, 51)");
        long personal = jdbc.queryForObject("SELECT id FROM afv_asset WHERE name = '个人全局资产'", Long.class);
        assertThat(queries.get(CanvasValueType.ASSET, personal, ACTOR).header().projectId()).isNull();
        rejects(() -> queries.get(CanvasValueType.ASSET, personal, new CanvasActor(52, null)), CanvasErrorCode.ACCESS_DENIED);
    }


    @Test
    void assetAndInitialVariantCreateTogetherWithTrustedOwnershipAndStableReplay() {
        ObjectNode fields = mapper.createObjectNode().put("type", "character").put("name", "角色")
                .put("coverUrl", "/media/images/character.png");
        fields.set("properties", mapper.createObjectNode().put("gender", "female"));
        var command = new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null, fields);
        var created = commands.execute(command, USER);
        assertThat(commands.execute(command, USER)).isEqualTo(created);
        assertThat(created.result().path("ownerType").intValue()).isEqualTo(1);
        assertThat(created.result().path("ownerId").longValue()).isEqualTo(51);
        assertThat(created.result().path("items").get(0).path("imageUrl").textValue()).isEqualTo("/media/images/character.png");
        assertThat(created.changes()).extracting(c -> c.resourceType()).containsExactly(CanvasValueType.ASSET, CanvasValueType.ASSET_ITEM);
        assertThat(created.changes()).allMatch(c -> c.version() == 0 && c.projectId() == null);
        long asset = created.result().path("id").longValue();
        rejects(() -> queries.get(CanvasValueType.ASSET, asset, new CanvasActor(52, null)), CanvasErrorCode.ACCESS_DENIED);
        long project = project();
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (?, 52, 3)", project);
        var collaborator = ResourceCommandContext.user(new CanvasActor(52, null), null);
        var projectAsset = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project,
                mapper.createObjectNode().put("type", "prop").put("name", "项目共享道具")), collaborator);
        assertThat(projectAsset.result().path("ownerId").longValue()).isEqualTo(51);
        assertThat(projectAsset.result().path("userId").longValue()).isEqualTo(52);
    }

    @Test
    void assetImageUpdateChecksParentVersionSoLateResultsCannotOverwriteChosenCover() {
        var created = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null,
                mapper.createObjectNode().put("type", "prop").put("name", "图像版本测试")), USER);
        long asset = created.result().path("id").longValue();
        long item = created.result().path("items").get(0).path("id").longValue();
        rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET_ITEM, item, 0,
                mapper.createObjectNode().put("imageUrl", "/media/images/missing-parent.png")), USER), CanvasErrorCode.INVALID_COMMAND);
        var update = new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET_ITEM, item, 0,
                mapper.createObjectNode().put("imageUrl", "/media/images/generated.png"), 0);
        var changed = commands.patch(update, USER);
        assertThat(commands.patch(update, USER)).isEqualTo(changed);
        assertThat(changed.changes()).hasSize(2);
        assertThat(queries.get(CanvasValueType.ASSET, asset, ACTOR).data().path("coverUrl").textValue()).isEqualTo("/media/images/generated.png");
        commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, asset, 1,
                mapper.createObjectNode().put("coverUrl", "/media/images/user-chosen.png")), USER);
        rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET_ITEM, item, 1,
                mapper.createObjectNode().put("imageUrl", "/media/images/late-result.png"), 1), USER), CanvasErrorCode.VERSION_CONFLICT);
        assertThat(queries.get(CanvasValueType.ASSET, asset, ACTOR).data().path("coverUrl").textValue()).isEqualTo("/media/images/user-chosen.png");
        assertThat(queries.get(CanvasValueType.ASSET_ITEM, item, ACTOR).data().path("assetVersion").intValue()).isEqualTo(2);
        assertThat(version("afv_asset_item", item)).isEqualTo(1);
    }

    @Test
    void assetCommandsValidateMediaPropertiesAndRollbackAllChildrenOnFailure() {
        for (ObjectNode fields : List.of(mapper.createObjectNode().put("type", "prop").put("name", "禁止内联")
                        .put("coverUrl", "data:image/png;base64,AA"),
                mapper.createObjectNode().put("type", "prop").put("name", "不能伪造").put("ownerId", 52))) {
            rejects(() -> commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null, fields), USER),
                    CanvasErrorCode.INVALID_COMMAND);
        }
        ObjectNode invalid = mapper.createObjectNode().put("type", "character").put("name", "无效选项");
        invalid.set("properties", mapper.createObjectNode().put("gender", "not-an-option"));
        rejects(() -> commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null, invalid), USER),
                CanvasErrorCode.INVALID_COMMAND);
        ObjectNode initial = mapper.createObjectNode().put("itemType", "generated_video");
        initial.set("properties", mapper.createObjectNode().put("sourceMediaUrl", "blob:browser-only"));
        ObjectNode nested = mapper.createObjectNode().put("type", "prop").put("name", "不能隐藏媒体");
        nested.set("initialItem", initial);
        rejects(() -> commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null, nested), USER),
                CanvasErrorCode.INVALID_COMMAND);
        var request = new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null,
                mapper.createObjectNode().put("type", "prop").put("name", "审计失败须回滚资产和初始图"));
        doThrow(new IllegalStateException("资产审计失败")).when(journal).commit(any(), any());
        assertThatThrownBy(() -> commands.execute(request, USER)).hasMessage("资产审计失败");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset WHERE name = '审计失败须回滚资产和初始图'", Integer.class)).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset_item WHERE name = '审计失败须回滚资产和初始图'", Integer.class)).isZero();
    }

    @Test
    void assetDeletionCascadesVersionsAndPreservesAnAuditedReplayForItsOwner() {
        var created = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null,
                mapper.createObjectNode().put("type", "scene").put("name", "完整资产删除")), USER);
        long asset = created.result().path("id").longValue();
        var item = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET_ITEM, null,
                mapper.createObjectNode().put("assetId", asset).put("parentVersion", 0).put("itemType", "variant").put("name", "夜晚")), USER);
        assertThat(item.result().path("assetVersion").intValue()).isEqualTo(1);
        var deletion = new ResourceDeleteCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, asset, 1);
        var result = commands.execute(deletion, USER);
        assertThat(commands.execute(deletion, USER)).isEqualTo(result);
        assertThat(result.changes()).hasSize(3).allMatch(c -> c.deleted());
        rejects(() -> queries.get(CanvasValueType.ASSET_ITEM, item.result().path("id").longValue(), ACTOR), CanvasErrorCode.RESOURCE_DELETED);
        assertThat(jdbc.queryForObject("SELECT undo_json FROM afv_resource_command WHERE id = ?", String.class, result.commandId())).contains("夜晚", "完整资产删除");
    }


    @Test
    void assetBatchCreatesOrExplicitlyReusesAtomicallyAndDoesNotPickAnAmbiguousName() {
        long project = project();
        var first = mapper.createObjectNode().put("type", "character").put("name", "角色甲");
        var second = mapper.createObjectNode().put("type", "scene").put("name", "客厅");
        var command = new AssetBatchCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project, null, null, true, List.of(first, second));
        var created = commands.execute(command, USER);
        assertThat(commands.execute(command, USER)).isEqualTo(created);
        assertThat(created.result().path("createdCount").intValue()).isEqualTo(2);
        assertThat(created.changes()).hasSize(4);
        var reused = commands.execute(new AssetBatchCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project, null, null, true, List.of(first)), USER);
        assertThat(reused.changes()).isEmpty();
        assertThat(reused.result().path("existingCount").intValue()).isEqualTo(1);
        var copy = commands.execute(new AssetBatchCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project, null, null, false, List.of(first)), USER);
        assertThat(copy.result().path("created").get(0).path("id").longValue()).isNotEqualTo(created.result().path("created").get(0).path("id").longValue());
        rejects(() -> commands.execute(new AssetBatchCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project, null, null, true, List.of(first)), USER),
                CanvasErrorCode.INVALID_COMMAND);
        var newAsset = mapper.createObjectNode().put("type", "prop").put("name", "不能部分提交");
        var invalid = mapper.createObjectNode().put("type", "scene").put("name", "无效").put("coverUrl", "data:image/png;base64,AA");
        rejects(() -> commands.execute(new AssetBatchCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project, null, null, false, List.of(newAsset, invalid)), USER),
                CanvasErrorCode.INVALID_COMMAND);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset WHERE project_id = ? AND name = '不能部分提交'", Integer.class, project)).isZero();
    }

    @Test
    void assetItemBatchReturnsCommittedParentVersionAndMergePreservesUneditedProperties() {
        var asset = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null,
                mapper.createObjectNode().put("type", "prop").put("name", "批量变体")), USER).result();
        long id = asset.path("id").longValue();
        var batch = new AssetBatchCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET_ITEM, null, id, 0, true,
                List.of(mapper.createObjectNode().put("name", "早晨"), mapper.createObjectNode().put("name", "夜晚")));
        var result = commands.execute(batch, USER);
        assertThat(result.result().path("assetVersion").intValue()).isEqualTo(2);
        for (var item : result.result().path("allItems")) assertThat(item.path("assetVersion").intValue()).isEqualTo(2);
        assertThat(commands.execute(batch, USER)).isEqualTo(result);
        var patch = mapper.createObjectNode();
        patch.set("properties", mapper.createObjectNode().put("material", "青铜").put("custom", "保留"));
        commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, id, 2, patch), USER);
        var merge = mapper.createObjectNode();
        merge.set("propertyChanges", mapper.createObjectNode().putNull("material").put("description", "新说明"));
        var changed = commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, id, 3, merge), USER).result();
        assertThat(changed.path("properties").path("custom").textValue()).isEqualTo("保留");
        assertThat(changed.path("properties").has("material")).isFalse();
        assertThat(changed.path("properties").path("description").textValue()).isEqualTo("新说明");
    }

    @Test
    void assetCollectionQueriesUseTheSameProjectAndGlobalOwnershipBoundariesAsDetails() {
        var assets = context.getBean(AssetResourceQueryService.class);
        long project = project();
        long hidden = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project,
                mapper.createObjectNode().put("type", "prop").put("name", "唯一权限筛选词")), USER).result().path("id").longValue();
        var other = new CanvasActor(52, null);
        assertThat(assets.list(null, null, "唯一权限筛选词", true, other)).isEmpty();
        rejects(() -> assets.list(project, null, null, false, other), CanvasErrorCode.ACCESS_DENIED);
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (?, 52, 3)", project);
        var page = assets.page(project, "prop", "唯一权限筛选词", 1, 20, other);
        assertThat(page.path("total").longValue()).isEqualTo(1);
        assertThat(page.path("records").get(0).path("id").longValue()).isEqualTo(hidden);
        assertThat(page.path("typeCounts").path("prop").longValue()).isEqualTo(1);
        assertThat(assets.list(project, null, null, true, other).get(0).path("items").size()).isEqualTo(1);
        assertThat(assets.list(project, null, "%", false, other)).isEmpty();
        jdbc.update("UPDATE afv_project_member SET deleted = TRUE WHERE project_id = ? AND user_id = 52", project);
        assertThat(assets.list(null, null, "唯一权限筛选词", false, other)).isEmpty();
        rejects(() -> assets.items(hidden, other), CanvasErrorCode.ACCESS_DENIED);
    }

    @Test
    void assetToolsUseObservedVersionsAndReturnCommittedChildIdentity() {
        long project = project();
        String run = agentRun(project);
        var json = context.getBean(CanvasJson.class);
        var tool = new AssetToolCommands(commands, queries, json);
        var invocation = ToolExecutionContext.forInvocation(51L, 1, 51L,
                new ResourceCommandContext(ACTOR, "agent", null, run, "create-asset-tool", "owner", 1L));
        String input = "{\"projectId\":" + project + ",\"type\":\"prop\",\"name\":\"工具创建\"}";
        var created = json.read(tool.execute("create_asset", input, invocation), ObjectNode.class);
        assertThat(created.path("status").textValue()).isEqualTo("success");
        assertThat(json.read(tool.execute("create_asset", input, invocation), ObjectNode.class)).isEqualTo(created);
        long item = created.path("items").get(0).path("assetItemId").longValue();
        var deletion = ToolExecutionContext.forInvocation(51L, 1, 51L,
                new ResourceCommandContext(ACTOR, "agent", null, run, "delete-item-tool", "owner", 1L));
        var deleted = json.read(tool.execute("delete_asset_resource", "{\"resourceType\":\"item\",\"resourceId\":" + item +
                ",\"expectedVersion\":0,\"expectedParentVersion\":0}", deletion), ObjectNode.class);
        assertThat(deleted.path("status").textValue()).isEqualTo("success");
        assertThat(deleted.path("assetItemId").longValue()).isEqualTo(item);
        assertThat(deleted.has("assetId")).isFalse();
    }

    private long script() {
        long project = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, null,
                mapper.createObjectNode().put("name", "剧本测试")), USER).result().path("id").longValue();
        return queries.workspace(project, ACTOR).path("script").path("id").longValue();
    }
    @Test
    void teamAssetCreationUsesPersistedTypedOwnershipAndRevokedMembersCannotReplay() {
        jdbc.update("INSERT INTO afv_team(id, name, owner_user_id) VALUES (9320, '团队资产空间', 52)");
        jdbc.update("INSERT INTO afv_team_member(team_id, user_id, role) VALUES (9320, 51, 3), (9320, 52, 1)");
        var actor = new CanvasActor(51, 9320L);
        var scope = ResourceCommandContext.user(actor, null);
        var command = new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null,
                mapper.createObjectNode().put("type", "prop").put("name", "团队共享资产"));
        var created = commands.execute(command, scope);
        long id = created.result().path("id").longValue();
        assertThat(created.result().path("ownerType").intValue()).isEqualTo(2);
        assertThat(created.result().path("ownerId").longValue()).isEqualTo(9320);
        assertThat(created.result().path("userId").longValue()).isEqualTo(51);
        assertThat(queries.get(CanvasValueType.ASSET, id, new CanvasActor(52, 9320L)).data().path("name").textValue()).isEqualTo("团队共享资产");
        rejects(() -> queries.get(CanvasValueType.ASSET, id, ACTOR), CanvasErrorCode.ACCESS_DENIED);
        jdbc.update("UPDATE afv_team_member SET status = 0 WHERE team_id = 9320 AND user_id = 51");
        rejects(() -> commands.execute(command, scope), CanvasErrorCode.ACCESS_DENIED);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset WHERE owner_type = 2 AND owner_id = 9320", Integer.class)).isEqualTo(1);
    }

    @Test
    void reusedBatchEntriesAndTheirInitialItemsStillRequireValidPropertiesAndAtomicity() {
        long project = project();
        ObjectNode existing = mapper.createObjectNode().put("type", "character").put("name", "已存在角色");
        commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project, existing), USER);
        ObjectNode badInitial = mapper.createObjectNode();
        badInitial.set("properties", mapper.createObjectNode().put("gender", "invalid"));
        ObjectNode reused = existing.deepCopy();
        reused.set("initialItem", badInitial);
        rejects(() -> commands.execute(new AssetBatchCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project, null, null, true,
                List.of(mapper.createObjectNode().put("type", "prop").put("name", "元数据失败不得残留"), reused)), USER), CanvasErrorCode.INVALID_COMMAND);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset WHERE project_id = ? AND name = '元数据失败不得残留'", Integer.class, project)).isZero();
        var valid = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project,
                mapper.createObjectNode().put("type", "character").put("name", "标准字段资产")), USER).result();
        ObjectNode child = mapper.createObjectNode().put("assetId", valid.path("id").longValue()).put("parentVersion", 0);
        child.set("properties", mapper.createObjectNode().put("gender", "invalid"));
        rejects(() -> commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET_ITEM, project, child), USER), CanvasErrorCode.INVALID_COMMAND);
        assertThat(version("afv_asset", valid.path("id").longValue())).isZero();
    }

    @Test
    void storyboardNestedCreationAndShotWritesShareVersionsAndNativeJson() {
        long board = storyboard();
        var command = new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null, storyboardDraft(board, 0, 1));
        Instant started = Instant.now().minusSeconds(1);
        var created = commands.execute(command, USER);
        assertThat(commands.execute(command, USER)).isEqualTo(created);
        assertThat(created.changes()).hasSize(5);
        long episode = created.result().path("id").longValue();
        assertThat(jdbc.queryForObject("SELECT create_time FROM afv_storyboard_episode WHERE id = ?",
                Timestamp.class, episode).toInstant()).isBetween(started, Instant.now().plusSeconds(1));
        var tree = context.getBean(StoryboardResourceQueryService.class).tree(CanvasValueType.STORYBOARD_EPISODE, episode, 2, ACTOR);
        long scene = tree.path("scenes").get(0).path("id").longValue();
        long shot = tree.path("scenes").get(0).path("items").get(0).path("id").longValue();
        assertThat(queries.get(CanvasValueType.STORYBOARD, board, ACTOR).data().path("totalDuration").intValue()).isEqualTo(9);
        var before = queries.get(CanvasValueType.SHOT, shot, ACTOR).data();
        assertThat(before.path("sceneVersion").intValue()).isZero();
        var fields = mapper.createObjectNode().put("firstFrameImageUrl", "https://example.com/frame.png").putNull("firstFramePrompt");
        fields.putArray("propIds");
        var patched = commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SHOT, shot, 0, fields), USER);
        assertThat(patched.result().path("propIds").isArray()).isTrue();
        assertThat(patched.result().path("firstFramePrompt").isNull()).isTrue();
        assertThat(patched.result().path("version").intValue()).isEqualTo(1);
        assertThat(patched.changes()).extracting(change -> change.resourceType()).containsExactly(
                CanvasValueType.SHOT, CanvasValueType.STORYBOARD_SCENE, CanvasValueType.STORYBOARD_EPISODE, CanvasValueType.STORYBOARD);
        assertThat(version("afv_storyboard_scene", scene)).isEqualTo(1);
        assertThat(version("afv_storyboard_episode", episode)).isEqualTo(1);
        assertThat(version("afv_storyboard", board)).isEqualTo(2);
        rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SHOT, shot, 0,
                mapper.createObjectNode().put("firstFrameImageUrl", "https://example.com/late.png")), USER), CanvasErrorCode.VERSION_CONFLICT);
        assertThat(queries.get(CanvasValueType.SHOT, shot, ACTOR).data().path("firstFrameImageUrl").textValue()).endsWith("frame.png");
    }

    @Test
    void storyboardCollectionsReorderAtomicallyAndClearCannotEraseAChangedChild() {
        long board = storyboard();
        var episode = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null, storyboardDraft(board, 0, 1)), USER).result();
        long episodeId = episode.path("id").longValue();
        var tree = context.getBean(StoryboardResourceQueryService.class).tree(CanvasValueType.STORYBOARD_EPISODE, episodeId, 2, ACTOR);
        long scene = tree.path("scenes").get(0).path("id").longValue();
        long first = tree.path("scenes").get(0).path("items").get(0).path("id").longValue();
        long second = tree.path("scenes").get(0).path("items").get(1).path("id").longValue();
        var order = new StoryboardChildrenCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_SCENE, scene, 0,
                StoryboardChildrenCommand.Mode.REORDER, List.of(), List.of(second, first));
        var reordered = commands.execute(order, USER);
        assertThat(commands.execute(order, USER)).isEqualTo(reordered);
        assertThat(reordered.result().path("children").get(0).path("id").longValue()).isEqualTo(second);
        assertThat(reordered.result().path("parent").path("version").intValue()).isEqualTo(1);
        rejects(() -> commands.execute(new StoryboardChildrenCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, episodeId, 0,
                StoryboardChildrenCommand.Mode.CLEAR, List.of(), List.of()), USER), CanvasErrorCode.VERSION_CONFLICT);
        rejects(() -> commands.execute(new StoryboardChildrenCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_SCENE, scene, 1,
                StoryboardChildrenCommand.Mode.REORDER, List.of(), List.of(first, first)), USER), CanvasErrorCode.INVALID_COMMAND);
        var clear = new StoryboardChildrenCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, episodeId, 1,
                StoryboardChildrenCommand.Mode.CLEAR, List.of(), List.of());
        var cleared = commands.execute(clear, USER);
        assertThat(commands.execute(clear, USER)).isEqualTo(cleared);
        assertThat(cleared.result().path("children")).isEmpty();
        assertThat(cleared.result().path("parent").path("version").intValue()).isEqualTo(2);
        rejects(() -> queries.get(CanvasValueType.SHOT, first, ACTOR), CanvasErrorCode.RESOURCE_DELETED);
        assertThat(queries.get(CanvasValueType.STORYBOARD, board, ACTOR).data().path("totalDuration").intValue()).isZero();
        var clearEmpty = commands.execute(new StoryboardChildrenCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, episodeId, 2,
                StoryboardChildrenCommand.Mode.CLEAR, List.of(), List.of()), USER);
        assertThat(clearEmpty.changes()).hasSize(2);
        assertThat(version("afv_storyboard", board)).isEqualTo(4);
    }

    @Test
    void storyboardBindsOnlyItsOwnScriptEpisodesAndReleasesBindingsOnDelete() {
        long board = storyboard();
        long script = queries.get(CanvasValueType.STORYBOARD, board, ACTOR).data().path("scriptId").longValue();
        long source = episode(script, 0, 1);
        ObjectNode fields = mapper.createObjectNode().put("storyboardId", board).put("parentVersion", 0).put("episodeNumber", 1).put("scriptEpisodeId", source);
        var first = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null, fields), USER).result();
        rejects(() -> commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null,
                fields.deepCopy().put("parentVersion", 1).put("episodeNumber", 2)), USER), CanvasErrorCode.INVALID_COMMAND);
        long otherScript = script();
        long foreign = episode(otherScript, 0, 1);
        rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, first.path("id").longValue(), 0,
                mapper.createObjectNode().put("scriptEpisodeId", foreign)), USER), CanvasErrorCode.INVALID_COMMAND);
        var deletion = new ResourceDeleteCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, first.path("id").longValue(), 0);
        var deleted = commands.execute(deletion, USER);
        assertThat(commands.execute(deletion, USER)).isEqualTo(deleted);
        var replacement = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null,
                fields.deepCopy().put("parentVersion", 2)), USER).result();
        assertThat(replacement.path("id").longValue()).isNotEqualTo(first.path("id").longValue());
        assertThat(replacement.path("scriptEpisodeId").longValue()).isEqualTo(source);
    }

    @Test
    void storyboardShotReferencesRejectCrossProjectWrongTypeAndUnsafeMediaWithoutPartialWrites() {
        long board = storyboard();
        long project = queries.get(CanvasValueType.STORYBOARD, board, ACTOR).header().projectId();
        var asset = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, project,
                mapper.createObjectNode().put("name", "可绑定道具").put("type", "prop")), USER).result();
        long prop = asset.path("items").get(0).path("id").longValue();
        var draft = storyboardDraft(board, 0, 1);
        var shot = (ObjectNode) draft.path("scenes").get(0).path("shots").get(0);
        shot.putArray("characterIds").add(prop);
        rejects(() -> commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null, draft), USER), CanvasErrorCode.INVALID_COMMAND);
        assertThat(version("afv_storyboard", board)).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_storyboard_episode WHERE storyboard_id = ?", Integer.class, board)).isZero();
        shot.remove("characterIds");
        shot.putArray("propIds").add(prop);
        var created = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null, draft), USER).result();
        long item = context.getBean(StoryboardResourceQueryService.class).tree(CanvasValueType.STORYBOARD_EPISODE, created.path("id").longValue(), 2, ACTOR)
                .path("scenes").get(0).path("items").get(0).path("id").longValue();
        assertThat(queries.get(CanvasValueType.SHOT, item, ACTOR).data().path("propIds").get(0).longValue()).isEqualTo(prop);
        var global = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null,
                mapper.createObjectNode().put("name", "尚未导入项目的角色").put("type", "character")), USER).result();
        ObjectNode foreign = mapper.createObjectNode();
        foreign.putArray("characterIds").add(global.path("items").get(0).path("id").longValue());
        rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SHOT, item, 0, foreign), USER), CanvasErrorCode.INVALID_COMMAND);
        for (ObjectNode invalid : List.of(mapper.createObjectNode().put("propIds", "[]"), mapper.createObjectNode().putNull("characterIds"),
                mapper.createObjectNode().put("videoUrl", "blob:not-durable"), mapper.createObjectNode().put("sortOrder", 3))) {
            rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SHOT, item, 0, invalid), USER), CanvasErrorCode.INVALID_COMMAND);
        }
        assertThat(version("afv_storyboard_item", item)).isZero();
    }

    @Test
    void storyboardBatchPatchChecksAllVersionsBeforeCommittingAndAuditsAtomically() {
        long board = storyboard();
        var episode = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null, storyboardDraft(board, 0, 1)), USER).result();
        var items = context.getBean(StoryboardResourceQueryService.class).tree(CanvasValueType.STORYBOARD_EPISODE, episode.path("id").longValue(), 2, ACTOR)
                .path("scenes").get(0).path("items");
        long first = items.get(0).path("id").longValue();
        long second = items.get(1).path("id").longValue();
        var valid = new StoryboardBatchPatchCommand.Entry(CanvasValueType.SHOT, first, 0, mapper.createObjectNode().put("content", "成批修改"));
        var stale = new StoryboardBatchPatchCommand.Entry(CanvasValueType.SHOT, second, 9, mapper.createObjectNode().put("content", "不可覆盖"));
        rejects(() -> commands.execute(new StoryboardBatchPatchCommand(UUID.randomUUID().toString(), board, List.of(valid, stale)), USER), CanvasErrorCode.VERSION_CONFLICT);
        assertThat(version("afv_storyboard_item", first)).isZero();
        var command = new StoryboardBatchPatchCommand(UUID.randomUUID().toString(), board, List.of(valid,
                new StoryboardBatchPatchCommand.Entry(CanvasValueType.SHOT, second, 0, mapper.createObjectNode().put("duration", 10))));
        doThrow(new IllegalStateException("分镜事务故障")).when(journal).commit(any(), any());
        assertThatThrownBy(() -> commands.execute(command, USER)).hasMessage("分镜事务故障");
        assertThat(version("afv_storyboard", board)).isEqualTo(1);
        assertThat(version("afv_storyboard_item", first)).isZero();
        reset(journal);
        var committed = commands.execute(command, USER);
        assertThat(commands.execute(command, USER)).isEqualTo(committed);
        assertThat(committed.changes()).hasSize(5);
        assertThat(queries.get(CanvasValueType.STORYBOARD, board, ACTOR).data().path("totalDuration").intValue()).isEqualTo(15);
        assertThat(committed.result().path("resources").get(0).path("data").path("sceneVersion").intValue()).isEqualTo(1);
    }

    @Test
    void storyboardDifferentShotWritersSerializeParentVersionsWithoutRejectingValidChildVersions() throws Exception {
        long board = storyboard();
        var episode = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null, storyboardDraft(board, 0, 1)), USER).result();
        long ep = episode.path("id").longValue();
        var items = context.getBean(StoryboardResourceQueryService.class).tree(CanvasValueType.STORYBOARD_EPISODE, ep, 2, ACTOR).path("scenes").get(0).path("items");
        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
            var first = pool.submit(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SHOT,
                    items.get(0).path("id").longValue(), 0, mapper.createObjectNode().put("content", "并发一")), USER));
            var second = pool.submit(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.SHOT,
                    items.get(1).path("id").longValue(), 0, mapper.createObjectNode().put("content", "并发二")), USER));
            assertThat(first.get(15, TimeUnit.SECONDS).result().path("version").intValue()).isEqualTo(1);
            assertThat(second.get(15, TimeUnit.SECONDS).result().path("version").intValue()).isEqualTo(1);
        }
        assertThat(version("afv_storyboard_episode", ep)).isEqualTo(2);
        assertThat(version("afv_storyboard", board)).isEqualTo(3);
    }

    @Test
    void storyboardToolsUseTrustedVersionsActualReceiptsAndCancelFencing() {
        long board = storyboard();
        var root = queries.get(CanvasValueType.STORYBOARD, board, ACTOR);
        long scriptEpisode = episode(root.data().path("scriptId").longValue(), 0, 1);
        String run = agentRun(root.header().projectId());
        var json = context.getBean(CanvasJson.class);
        var tools = new StoryboardToolCommands(commands, queries, json);
        var reads = new StoryboardToolQueries(context.getBean(StoryboardResourceQueryService.class), queries, mock(SystemConfigService.class), json);
        ObjectNode creation = mapper.createObjectNode().put("storyboardId", board).put("scriptEpisodeId", scriptEpisode)
                .put("episodeNumber", 1).put("title", "工具分镜集").put("expectedParentVersion", 0);
        var episode = json.read(tools.execute("save_storyboard_episode", json.write(creation), storyboardInvocation(run, "create-episode")), ObjectNode.class);
        assertThat(episode.path("status").textValue()).isEqualTo("success");
        assertThat(episode.path("commandId").asText()).isNotBlank();
        assertThat(json.read(tools.execute("save_storyboard_episode", json.write(creation), storyboardInvocation(run, "create-episode")), ObjectNode.class)).isEqualTo(episode);
        ObjectNode scene = mapper.createObjectNode().put("storyboardId", board).put("storyboardEpisodeId", episode.path("storyboardEpisodeId").longValue())
                .put("expectedParentVersion", 0).put("sceneNumber", "1-1");
        scene.putArray("shots").addObject().put("content", "真实工具镜头");
        var created = json.read(tools.execute("save_storyboard_scene_shots", json.write(scene), storyboardInvocation(run, "create-scene")), ObjectNode.class);
        assertThat(created.path("status").textValue()).isEqualTo("success");
        assertThat(created.path("episodeVersion").intValue()).isEqualTo(1);
        long item = created.path("createdShotIds").get(0).longValue();
        var details = json.read(reads.execute("get_storyboard_scene_items", json.write(mapper.createObjectNode().put("storyboardItemId", item)),
                storyboardInvocation(run, "read-shot")), ObjectNode.class);
        assertThat(details.path("items").get(0).path("version").intValue()).isZero();
        assertThat(details.path("sceneVersion").intValue()).isZero();
        var frame = mapper.createObjectNode().put("storyboardItemId", item).put("expectedVersion", 0)
                .put("frameType", "first").put("imageUrl", "https://example.com/frame.png");
        var bound = json.read(tools.execute("update_storyboard_item_frame", json.write(frame), storyboardInvocation(run, "bind-frame")), ObjectNode.class);
        assertThat(bound.path("status").textValue()).isEqualTo("success");
        assertThat(bound.path("version").intValue()).isEqualTo(1);
        var video = mapper.createObjectNode().put("storyboardItemId", item).put("expectedVersion", 0)
                .put("videoUrl", "https://example.com/video.mp4").put("coverUrl", "https://example.com/poster.png");
        var rejected = json.read(tools.execute("update_storyboard_item_video", json.write(video), storyboardInvocation(run, "stale-video")), ObjectNode.class);
        assertThat(rejected.path("errorCode").textValue()).isEqualTo("VERSION_CONFLICT");
        video.put("expectedVersion", 1);
        var saved = json.read(tools.execute("update_storyboard_item_video", json.write(video), storyboardInvocation(run, "bind-video")), ObjectNode.class);
        assertThat(saved.path("status").textValue()).isEqualTo("success");
        assertThat(saved.path("customData").path("videoPosterUrl").textValue()).endsWith("poster.png");
        jdbc.update("UPDATE afv_agent_run SET status = 'CANCEL_REQUESTED', cancel_requested_at = UTC_TIMESTAMP(3) WHERE run_id = ?", run);
        var cancelled = json.read(tools.execute("update_storyboard_item", json.write(mapper.createObjectNode().put("storyboardItemId", item)
                .put("expectedVersion", 2).put("content", "取消后不可写")), storyboardInvocation(run, "late-write")), ObjectNode.class);
        assertThat(cancelled.path("errorCode").textValue()).isEqualTo("ACCESS_DENIED");
        assertThat(queries.get(CanvasValueType.SHOT, item, ACTOR).data().path("content").textValue()).isEqualTo("真实工具镜头");
    }

    @Test
    void storyboardGlobalShotQueriesFollowEpisodeSceneAndShotOrder() {
        long board = storyboard();
        ObjectNode second = storyboardDraft(board, 0, 2);
        ((ObjectNode) second.path("scenes").get(0).path("shots").get(0)).put("content", "2A");
        ((ObjectNode) second.path("scenes").get(0).path("shots").get(1)).put("content", "2B");
        commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null, second), USER);
        ObjectNode first = storyboardDraft(board, 1, 1).put("sortOrder", 0);
        ((ObjectNode) first.path("scenes").get(0).path("shots").get(0)).put("content", "1A");
        ((ObjectNode) first.path("scenes").get(0).path("shots").get(1)).put("content", "1B");
        commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD_EPISODE, null, first), USER);
        var list = context.getBean(StoryboardResourceQueryService.class).children(CanvasValueType.SHOT, CanvasValueType.STORYBOARD, board, ACTOR);
        assertThat(list).extracting(item -> item.path("content").textValue()).containsExactly("1A", "1B", "2A", "2B");
        var statistics = context.getBean(StoryboardResourceQueryService.class).statistics(board, ACTOR);
        assertThat(statistics.episodeCount()).isEqualTo(2);
        assertThat(statistics.sceneCount()).isEqualTo(2);
        assertThat(statistics.itemCount()).isEqualTo(4);
    }

    private ToolExecutionContext storyboardInvocation(String run, String call) {
        return ToolExecutionContext.forInvocation(51L, 1, 51L, new ResourceCommandContext(ACTOR, "agent", null, run, call, "owner", 1L));
    }

    @Test
    void storyboardRootIdentityAndLifecycleCannotBeBypassedByGenericResourceCommands() {
        long board = storyboard();
        rejects(() -> commands.execute(new ResourceDeleteCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD, board, 0), USER), CanvasErrorCode.INVALID_COMMAND);
        for (String field : List.of("projectId", "scriptId", "ownerId", "title")) {
            ObjectNode patch = mapper.createObjectNode().put(field, "不可绕过的归属或标题");
            rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.STORYBOARD, board, 0, patch), USER), CanvasErrorCode.INVALID_COMMAND);
        }
        assertThat(queries.get(CanvasValueType.STORYBOARD, board, ACTOR).header().version()).isZero();
    }

    @Test
    void teamVisibilityDoesNotGrantProjectOrPersonalAssetWritesAndRevokedScopeCannotReplay() {
        jdbc.update("INSERT INTO afv_team(id, name, owner_user_id) VALUES (9340, '只读团队可见性', 51)");
        jdbc.update("INSERT INTO afv_team_member(team_id, user_id, role) VALUES (9340, 51, 1), (9340, 52, 3)");
        long project = project();
        var actor = new CanvasActor(52, 9340L);
        var member = ResourceCommandContext.user(actor, null);
        assertThat(queries.get(CanvasValueType.PROJECT, project, actor).header().resourceId()).isEqualTo(project);
        var change = command(project, 0, mapper.createObjectNode().put("description", "明确授权成员修改"));
        rejects(() -> commands.patch(change, member), CanvasErrorCode.ACCESS_DENIED);
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (?, 52, 3)", project);
        var committed = commands.patch(change, member);
        assertThat(committed.result().path("version").intValue()).isEqualTo(1);
        jdbc.update("UPDATE afv_project_member SET deleted = TRUE WHERE project_id = ? AND user_id = 52", project);
        rejects(() -> commands.patch(change, member), CanvasErrorCode.ACCESS_DENIED);
        var personal = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, null,
                mapper.createObjectNode().put("type", "prop").put("name", "个人可见素材")), USER).result();
        long asset = personal.path("id").longValue();
        assertThat(queries.get(CanvasValueType.ASSET, asset, actor).header().resourceId()).isEqualTo(asset);
        rejects(() -> commands.patch(new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.ASSET, asset, 0,
                mapper.createObjectNode().put("name", "不能修改同事个人素材")), member), CanvasErrorCode.ACCESS_DENIED);
        var ownScope = ResourceCommandContext.user(new CanvasActor(51, 9340L), null);
        jdbc.update("UPDATE afv_team_member SET status = 0 WHERE team_id = 9340 AND user_id = 51");
        rejects(() -> commands.patch(command(project, 1, mapper.createObjectNode().put("description", "已失效团队身份")), ownScope), CanvasErrorCode.ACCESS_DENIED);
        rejects(() -> queries.get(CanvasValueType.PROJECT, project, ownScope.actor()), CanvasErrorCode.ACCESS_DENIED);
        assertThat(queries.get(CanvasValueType.PROJECT, project, ACTOR).header().version()).isEqualTo(1);
    }

    private long storyboard() {
        long project = commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, null,
                mapper.createObjectNode().put("name", "分镜项目-" + UUID.randomUUID())), USER).result().path("id").longValue();
        return queries.workspace(project, ACTOR).path("storyboard").path("id").longValue();
    }
    private ObjectNode storyboardDraft(long board, int version, int number) {
        var root = mapper.createObjectNode().put("storyboardId", board).put("parentVersion", version).put("episodeNumber", number).put("title", "测试集");
        var shots = root.putArray("scenes").addObject().put("sceneNumber", number + "-1").put("sceneHeading", "内景 客厅 夜").putArray("shots");
        shots.addObject().put("content", "镜头甲").put("duration", 4.25);
        shots.addObject().put("content", "镜头乙").put("duration", 4.25);
        return root;
    }

    private long episode(long script, int version, int number) {
        return commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_EPISODE, null,
                mapper.createObjectNode().put("scriptId", script).put("parentVersion", version).put("episodeNumber", number)), USER).result().path("id").longValue();
    }
    private long scene(long episode, int version, String heading) {
        return commands.execute(new ResourceCreateCommand(UUID.randomUUID().toString(), CanvasValueType.SCRIPT_SCENE, null,
                mapper.createObjectNode().put("episodeId", episode).put("parentVersion", version).put("sceneHeading", heading)), USER).result().path("id").longValue();
    }

    @Test
    void resourceFeedReadsCommittedChangesWithoutCanvasAndChecksCurrentPermissions() {
        long project = project();
        drain();
        var feed = context.getBean(ResourceChangeFeed.class);
        var ownerStart = feed.read(ACTOR, null, null, 200);
        var outsider = new CanvasActor(52, null);
        var otherStart = feed.read(outsider, null, null, 200);
        var result = commands.patch(command(project, 0, mapper.createObjectNode().put("name", "跨入口变化")), USER);
        assertThat(feed.read(ACTOR, ownerStart.cursor(), ownerStart.scope(), 200).events()).isEmpty();
        drain();
        var ownerPage = feed.read(ACTOR, ownerStart.cursor(), ownerStart.scope(), 200);
        assertThat(ownerPage.events()).anySatisfy(entry -> {
            assertThat(entry.commandId()).isEqualTo(result.commandId());
            assertThat(entry.change().resourceId()).isEqualTo(project);
            assertThat(entry.change().version()).isEqualTo(1);
        });
        var otherPage = feed.read(outsider, otherStart.cursor(), otherStart.scope(), 200);
        assertThat(otherPage.events()).isEmpty();
        assertThat(otherPage.cursor()).isEqualTo(ownerPage.cursor());
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (?, 52, 3)", project);
        var granted = feed.read(outsider, otherPage.cursor(), otherPage.scope(), 200);
        assertThat(granted.reset()).isTrue();
        jdbc.update("UPDATE afv_project_member SET deleted = TRUE WHERE project_id = ? AND user_id = 52", project);
        assertThat(feed.read(outsider, granted.cursor(), granted.scope(), 200).reset()).isTrue();
        rejects(() -> feed.read(ACTOR, ownerPage.head() + 1, ownerPage.scope(), 200), CanvasErrorCode.INVALID_CURSOR);
    }

    @Test
    void resourceFeedDoesNotLoseLowerIdTransactionsThatCommitLater() throws Exception {
        long firstProject = project();
        long secondProject = project();
        drain();
        var feed = context.getBean(ResourceChangeFeed.class);
        var start = feed.read(ACTOR, null, null, 200);
        var early = command(firstProject, 0, mapper.createObjectNode().put("name", "较早编号较晚提交"));
        var inserted = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        doAnswer(call -> {
            call.callRealMethod();
            ResourceCommandReceipt receipt = call.getArgument(0);
            if (receipt.requestId().equals(early.requestId())) {
                inserted.countDown();
                if (!release.await(20, TimeUnit.SECONDS)) throw new IllegalStateException("未释放测试事务");
            }
            return null;
        }).when(journal).commit(any(), any());
        try (var pool = Executors.newSingleThreadExecutor()) {
            var pending = pool.submit(() -> commands.patch(early, USER));
            try {
                assertThat(inserted.await(10, TimeUnit.SECONDS)).isTrue();
                var late = commands.patch(command(secondProject, 0, mapper.createObjectNode().put("name", "较晚编号先提交")), USER);
                drain();
                var first = feed.read(ACTOR, start.cursor(), start.scope(), 200);
                assertThat(first.events()).extracting(ResourceChangeFeed.Entry::commandId).containsExactly(late.commandId());
                release.countDown();
                var earlyResult = pending.get(20, TimeUnit.SECONDS);
                drain();
                var second = feed.read(ACTOR, first.cursor(), first.scope(), 200);
                assertThat(second.events()).extracting(ResourceChangeFeed.Entry::commandId).containsExactly(earlyResult.commandId());
                assertThat(second.cursor()).isGreaterThan(first.cursor());
            } finally { release.countDown(); }
        }
    }

    private void drain() { while (projection.publishNext()) { } }
    private ResourcePatchCommand command(long id, int version, ObjectNode patch) { return new ResourcePatchCommand(UUID.randomUUID().toString(), CanvasValueType.PROJECT, id, version, patch); }
    private int version(String table, long id) { return jdbc.queryForObject("SELECT version FROM " + table + " WHERE id = ?", Integer.class, id); }
    private long project() {
        String name = UUID.randomUUID().toString();
        jdbc.update("INSERT INTO afv_project(name, owner_type, owner_id) VALUES (?, 1, 51)", name);
        return jdbc.queryForObject("SELECT id FROM afv_project WHERE name = ?", Long.class, name);
    }
    private String canvas(long project) {
        String id = UUID.randomUUID().toString();
        context.getBean(CanvasDocumentRepository.class).admit(id, "项目画布", project, 1, ACTOR.userId(), ACTOR.userId());
        return id;
    }
    private void rejects(Runnable action, CanvasErrorCode code) {
        assertThatThrownBy(action::run).isInstanceOfSatisfying(CanvasException.class, failure -> assertThat(failure.getErrorCode()).isEqualTo(code));
    }
    private String agentRun(long project) {
        String run = UUID.randomUUID().toString();
        jdbc.update("""
                INSERT INTO afv_agent_run(run_id, conversation_id, user_id, project_id, kernel_fingerprint,
                    agent_definition_snapshot_json, agent_state_session_id, status, owner_instance_id, owner_epoch,
                    lease_until, deadline_at, started_at)
                VALUES (?, ?, 51, ?, 'test', '{}', ?, 'RUNNING', 'owner', 1,
                    DATE_ADD(UTC_TIMESTAMP(3), INTERVAL 10 MINUTE), DATE_ADD(UTC_TIMESTAMP(3), INTERVAL 1 HOUR), UTC_TIMESTAMP(3))
                """, run, run, project, run);
        return run;
    }
}
