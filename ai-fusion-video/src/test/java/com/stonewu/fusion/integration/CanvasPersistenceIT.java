package com.stonewu.fusion.integration;

import com.stonewu.fusion.mapper.canvas.CanvasNodeMapper;
import com.stonewu.fusion.mapper.canvas.CanvasEdgeMapper;
import com.stonewu.fusion.mapper.canvas.CanvasOperationMapper;
import com.stonewu.fusion.mapper.canvas.CanvasEventMapper;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.config.CacheConfig;
import com.stonewu.fusion.service.resource.ResourceAgentFence;
import com.stonewu.fusion.repository.canvas.CanvasAccessRepository;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.repository.canvas.CanvasJournalRepository;
import com.stonewu.fusion.repository.canvas.CanvasWorkspaceRepository;
import com.stonewu.fusion.repository.resource.ProjectWorkspaceRepository;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceSnapshotCache;
import com.stonewu.fusion.service.resource.StoryboardVersionView;
import com.stonewu.fusion.service.resource.ProjectFields;
import com.stonewu.fusion.service.canvas.CanvasUndoService;
import com.stonewu.fusion.service.canvas.CanvasWorkspaceService;
import com.stonewu.fusion.service.canvas.command.CanvasUndoEngine;
import com.stonewu.fusion.service.canvas.command.CanvasUndoCommand;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.CommentCreate;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.CommentPatch;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.CheckpointCreate;
import com.stonewu.fusion.service.canvas.command.CanvasWorkspaceCommands.DocumentPatch;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasCommandService;
import com.stonewu.fusion.service.canvas.CanvasEventStream;
import com.stonewu.fusion.service.canvas.CanvasGraphValidator;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.CanvasRequestValidator;
import com.stonewu.fusion.service.canvas.CanvasSnapshotCache;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasConfigValidator;
import com.stonewu.fusion.service.canvas.command.CanvasCommandReceipt;
import com.stonewu.fusion.service.canvas.command.CanvasCreateCommand;
import com.stonewu.fusion.service.canvas.command.CanvasEditCommand;
import com.stonewu.fusion.service.canvas.command.CanvasEditEngine;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.CreateEdge;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.CreateNode;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.DeleteNode;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.LayoutNode;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.PatchNode;
import com.stonewu.fusion.service.canvas.command.CanvasGraphOperation.ReplaceEdge;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.model.CanvasEdge;
import com.stonewu.fusion.service.canvas.model.CanvasEdgeKind;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasNodeLayout;
import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;
import com.stonewu.fusion.service.canvas.model.CanvasSnapshot;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import jakarta.validation.ValidatorFactory;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.cache.CacheManager;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Configuration;
import org.springframework.transaction.annotation.EnableTransactionManagement;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.jdbc.support.GeneratedKeyHolder;
import org.springframework.transaction.PlatformTransactionManager;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import reactor.test.StepVerifier;

import java.sql.Statement;
import java.time.Duration;
import java.util.ArrayList;

import com.stonewu.fusion.service.canvas.CanvasBindingPolicy;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.spy;

@Testcontainers(disabledWithoutDocker = false)
class CanvasPersistenceIT {
    @Configuration
    @EnableTransactionManagement
    static class Transactions {
    }

    @Container
    private static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("canvas_commands_test").withUsername("afv").withPassword("afv-test");
    @Container
    private static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(
            6379);
    private static final CanvasActor OWNER = new CanvasActor(101, null);
    private static final CanvasActor OTHER = new CanvasActor(102, null);
    private static AnnotationConfigApplicationContext context;
    private static ValidatorFactory validation;
    private static JdbcTemplate jdbc;
    private static CanvasCommandService commands;
    private static CanvasQueryService queries;
    private static CanvasGraphRepository graphs;
    private static CanvasJournalRepository journal;
    private static ObjectMapper mapper;

    @BeforeAll
    static void start() throws Exception {
        var source = new DriverManagerDataSource(MYSQL.getJdbcUrl(), MYSQL.getUsername(), MYSQL.getPassword());
        Flyway flyway = Flyway.configure().dataSource(source).load();
        flyway.migrate();
        flyway.validate();
        jdbc = new JdbcTemplate(source);
        jdbc.update(
                "INSERT INTO sys_user(id, username, password) VALUES (101, 'canvas-owner', 'test-only'), (102, 'canvas-member', 'test-only')");
        mapper = new ObjectMapper().findAndRegisterModules();
        CanvasJson json = new CanvasJson(mapper);
        context = new AnnotationConfigApplicationContext();
        var session = MybatisTestSupport.register(context, source);
        graphs = spy(new CanvasGraphRepository(session.getMapper(CanvasNodeMapper.class),
                session.getMapper(CanvasEdgeMapper.class), json));
        journal = spy(new CanvasJournalRepository(session.getMapper(CanvasOperationMapper.class),
                session.getMapper(CanvasEventMapper.class), json));
        validation = Validation.buildDefaultValidatorFactory();
        context.registerBean(JdbcTemplate.class, () -> jdbc);
        context.registerBean(PlatformTransactionManager.class, () -> new DataSourceTransactionManager(source));
        context.registerBean(ObjectMapper.class, () -> mapper);
        context.registerBean(Validator.class, validation::getValidator);
        context.registerBean(CanvasGraphRepository.class, () -> graphs);
        context.registerBean(CanvasJournalRepository.class, () -> journal);
        context.registerBean(RedisConnectionFactory.class,
                () -> new LettuceConnectionFactory(REDIS.getHost(), REDIS.getMappedPort(6379)));
        context.register(CacheConfig.class, CanvasJson.class, CanvasDocumentRepository.class,
                CanvasAccessRepository.class,
                CanvasAccessPolicy.class, CanvasSnapshotCache.class, CanvasRequestValidator.class,
                CanvasQueryService.class,
                CanvasCommandService.class, ResourceAgentFence.class, CanvasEditEngine.class,
                CanvasGraphValidator.class, CanvasCapabilityRegistry.class,
                CanvasConfigValidator.class, CanvasEventStream.class, CanvasBindingPolicy.class,
                ResourceReadRepository.class, ResourceAccessPolicy.class,
                Transactions.class, CanvasUndoEngine.class, CanvasUndoService.class, CanvasWorkspaceService.class,
                CanvasWorkspaceRepository.class,
                ResourceQueryService.class, ResourceSnapshotCache.class, ProjectWorkspaceRepository.class,
                StoryboardVersionView.class, ProjectFields.class);
        context.refresh();
        commands = context.getBean(CanvasCommandService.class);
        queries = context.getBean(CanvasQueryService.class);
    }

    @AfterEach
    void clearFaults() {
        reset(graphs, journal);
    }

    @AfterAll
    static void close() {
        if (context != null) {
            context.close();
        }
        if (validation != null) {
            validation.close();
        }
    }

    @Test
    void createsIdempotentlyAndCachesVersionedSnapshotsUsingProductionRedisSerializer() {
        CanvasCreateCommand create = new CanvasCreateCommand(id(), "创作画布", null);
        CanvasDocument document = commands.create(create, OWNER);
        assertThat(commands.create(create, OWNER)).isEqualTo(document);
        rejects(() -> commands.create(new CanvasCreateCommand(create.id(), "不同内容", null), OWNER),
                CanvasErrorCode.REQUEST_CONFLICT);
        rejects(() -> commands.create(create, OTHER), CanvasErrorCode.ACCESS_DENIED);
        CanvasNode text = node(CanvasNodeKind.TEXT, mapper.createObjectNode().put("content", "内容"));
        CanvasEditCommand command = command(new CreateNode(text));
        CanvasCommandReceipt receipt = commands.edit(document.id(), command, OWNER);
        assertThat(commands.edit(document.id(), command, OWNER)).isEqualTo(receipt);
        CanvasSnapshot snapshot = queries.snapshot(document.id(), OWNER);
        assertThat(snapshot.graph().nodes()).containsExactly(text);
        assertThat(snapshot.lastSequence()).isEqualTo(2);
        assertThat(snapshot.document().revision()).isEqualTo(2);
        var cache = context.getBean(CacheManager.class).getCache("canvasGraph");
        assertThat(Objects.requireNonNull(cache).get(document.id() + ":2", String.class)).contains("内容");
        assertThat(queries.snapshot(document.id(), OWNER)).isEqualTo(snapshot);
        assertThat(count("afv_canvas_operation", document.id())).isEqualTo(2);
        assertThat(count("afv_canvas_event", document.id())).isEqualTo(2);
    }

    @Test
    void concurrentlyReplaysOneReceiptWithoutDuplicatingNodesOrEvents() throws Exception {
        String canvas = create();
        CanvasEditCommand command = command(new CreateNode(text()));
        CountDownLatch start = new CountDownLatch(1);
        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<CanvasCommandReceipt>> results = new ArrayList<>();
            for (int index = 0; index < 8; index++) {
                results.add(pool.submit(() -> {
                    assertThat(start.await(5, TimeUnit.SECONDS)).isTrue();
                    return commands.edit(canvas, command, OWNER);
                }));
            }
            start.countDown();
            CanvasCommandReceipt first = results.getFirst().get(20, TimeUnit.SECONDS);
            for (Future<CanvasCommandReceipt> result : results) {
                assertThat(result.get(20, TimeUnit.SECONDS)).isEqualTo(first);
            }
        }
        assertThat(count("afv_canvas_node", canvas)).isEqualTo(1);
        assertThat(count("afv_canvas_operation", canvas)).isEqualTo(2);
        assertThat(count("afv_canvas_event", canvas)).isEqualTo(2);
    }

    @Test
    void independentConfigurationAndLayoutWritesSurviveConcurrencyButConflictingFieldsDoNot() throws Exception {
        String canvas = create();
        CanvasNode node = text();
        edit(canvas, new CreateNode(node));
        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
            var moved = pool.submit(() -> edit(canvas, new LayoutNode(node.id(), 0, layout(80, 120, null))));
            var renamed = pool.submit(() -> edit(canvas, new PatchNode(node.id(), 0, "改名", null)));
            moved.get(20, TimeUnit.SECONDS);
            renamed.get(20, TimeUnit.SECONDS);
            var one = pool.submit(() -> patchResult(canvas, node.id(), "写入一"));
            var two = pool.submit(() -> patchResult(canvas, node.id(), "写入二"));
            assertThat(List.of(one.get(20, TimeUnit.SECONDS), two.get(20, TimeUnit.SECONDS)))
                    .containsExactlyInAnyOrder("committed", "VERSION_CONFLICT");
        }
        CanvasNode result = queries.snapshot(canvas, OWNER).graph().nodes().getFirst();
        assertThat(result.layout().x()).isEqualTo(80);
        assertThat(result.layoutVersion()).isEqualTo(1);
        assertThat(result.configVersion()).isEqualTo(2);
        assertThat(count("afv_canvas_event", canvas)).isEqualTo(5);
    }

    @Test
    void journalFailureRollsBackAlreadyWrittenGraphAndDoesNotAdvanceSequence() {
        String canvas = create();
        doThrow(new IllegalStateException("模拟事件存储失败")).when(journal)
                .append(anyLong(), anyString(), isNull(), isNull(),
                        anyString(), any(), any(), any(), anyString());
        assertThatThrownBy(() -> edit(canvas, new CreateNode(text()))).hasMessage("模拟事件存储失败");
        assertThat(count("afv_canvas_node", canvas)).isZero();
        assertThat(count("afv_canvas_operation", canvas)).isEqualTo(1);
        assertThat(queries.snapshot(canvas, OWNER).lastSequence()).isEqualTo(1);
    }

    @Test
    void rejectsGlobalIdentifierReuseWithoutPartiallyWritingAnotherNode() {
        String first = create();
        String second = create();
        CanvasNode node = text();
        edit(first, new CreateNode(node));
        rejects(() -> edit(second, new CreateNode(text()), new CreateNode(node)), CanvasErrorCode.REQUEST_CONFLICT);
        assertThat(count("afv_canvas_node", second)).isZero();
        assertThat(count("afv_canvas_event", second)).isEqualTo(1);
    }

    @Test
    void reordersOccupiedInputsAtomicallyAndDetachesChildrenWhenDeletingGroups() {
        String canvas = create();
        CanvasNode group = node(CanvasNodeKind.GROUP, mapper.createObjectNode());
        group = new CanvasNode(group.id(), group.kind(), 1, group.title(), layout(100, 200, null), group.config(), null,
                0, 0);
        CanvasNode child = text();
        child = new CanvasNode(child.id(), child.kind(), 1, child.title(), layout(5, 10, group.id()), child.config(),
                null, 0, 0);
        CanvasNode other = text();
        CanvasNode collection = node(CanvasNodeKind.COLLECTION, mapper.createObjectNode().put("valueType", "text"));
        CanvasEdge one = edge(child.id(), collection.id(), 0);
        CanvasEdge two = edge(other.id(), collection.id(), 1);
        edit(canvas, new CreateNode(child), new CreateNode(group), new CreateNode(other), new CreateNode(collection),
                new CreateEdge(one), new CreateEdge(two));
        edit(canvas, new ReplaceEdge(at(one, 1), 0), new ReplaceEdge(at(two, 0), 0), new DeleteNode(group.id(), 0, 0));
        CanvasSnapshot snapshot = queries.snapshot(canvas, OWNER);
        String childId = child.id();
        CanvasNode result = snapshot.graph().nodes().stream().filter(item -> item.id().equals(childId)).findFirst()
                .orElseThrow();
        assertThat(result.layout()).isEqualTo(layout(105, 210, null));
        assertThat(snapshot.graph().edges()).extracting(CanvasEdge::inputIndex).containsExactlyInAnyOrder(0, 1);
        assertThat(snapshot.graph().edges()).allMatch(item -> item.version() == 1);
    }

    @Test
    void staleSnapshotCacheWriteCannotOverwriteNewRevision() throws Exception {
        String canvas = create();
        CanvasNode node = text();
        edit(canvas, new CreateNode(node));
        CountDownLatch loaded = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        AtomicBoolean pause = new AtomicBoolean(true);
        doAnswer(invocation -> {
            Object graph = invocation.callRealMethod();
            CanvasDocument document = invocation.getArgument(0);
            if (document.id().equals(canvas) && document.revision() == 2 && pause.compareAndSet(true, false)) {
                loaded.countDown();
                assertThat(release.await(10, TimeUnit.SECONDS)).isTrue();
            }
            return graph;
        }).when(graphs).load(any());
        try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
            var old = pool.submit(() -> queries.snapshot(canvas, OWNER));
            try {
                assertThat(loaded.await(10, TimeUnit.SECONDS)).isTrue();
                edit(canvas, new PatchNode(node.id(), 0, "最新标题", null));
            } finally {
                release.countDown();
            }
            assertThat(old.get(10, TimeUnit.SECONDS).document().revision()).isEqualTo(2);
        }
        CanvasSnapshot latest = queries.snapshot(canvas, OWNER);
        assertThat(latest.document().revision()).isEqualTo(3);
        assertThat(latest.graph().nodes().getFirst().title()).isEqualTo("最新标题");
    }

    @Test
    void editsEvictOnlyThePreviousRevisionOfTheAffectedCanvas() {
        String first = create();
        String second = create();
        queries.snapshot(first, OWNER);
        queries.snapshot(second, OWNER);
        edit(first, new CreateNode(text()));
        var cache = Objects.requireNonNull(context.getBean(CacheManager.class).getCache("canvasGraph"));
        assertThat(cache.get(first + ":1")).isNull();
        assertThat(cache.get(second + ":1")).isNotNull();
    }

    @Test
    void distinguishesViewEditAndExecuteAndRechecksRevocationBeforeReplay() {
        String canvas = create();
        share(canvas, OTHER.userId(), false, true);
        CanvasSnapshot view = queries.snapshot(canvas, OTHER);
        assertThat(view.permissions().canView()).isTrue();
        assertThat(view.permissions().canExecute()).isTrue();
        assertThat(view.permissions().canEdit()).isFalse();
        rejects(() -> commands.edit(canvas, command(new CreateNode(text())), OTHER), CanvasErrorCode.ACCESS_DENIED);
        CanvasEditCommand edit = command(new CreateNode(text()));
        jdbc.update("UPDATE afv_canvas_member SET can_edit = TRUE WHERE canvas_id = ?", canvas);
        commands.edit(canvas, edit, OTHER);
        jdbc.update("DELETE FROM afv_canvas_member WHERE canvas_id = ?", canvas);
        rejects(() -> queries.snapshot(canvas, OTHER), CanvasErrorCode.ACCESS_DENIED);
        rejects(() -> queries.events(canvas, OTHER, 0, 10), CanvasErrorCode.ACCESS_DENIED);
        rejects(() -> commands.edit(canvas, edit, OTHER), CanvasErrorCode.ACCESS_DENIED);
    }

    @Test
    void projectRevocationBlocksEvenCanvasCreatorAndProjectManagerCanStillManage() {
        long project = project(OWNER.userId());
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (?, ?, 3)", project,
                OTHER.userId());
        String canvas = commands.create(new CanvasCreateCommand(id(), "成员画布", project), OTHER).id();
        assertThat(queries.list(OTHER, project, "", false, 1, 100).getList()).extracting(item -> item.document().id())
                .contains(canvas);
        assertThat(queries.snapshot(canvas, OWNER).permissions().canManage()).isTrue();
        queries.snapshot(canvas, OTHER);
        jdbc.update("UPDATE afv_project_member SET deleted = TRUE WHERE project_id = ? AND user_id = ?", project,
                OTHER.userId());
        rejects(() -> queries.snapshot(canvas, OTHER), CanvasErrorCode.ACCESS_DENIED);
        assertThat(queries.list(OTHER, null, "", false, 1, 100).getList()).extracting(item -> item.document().id())
                .doesNotContain(canvas);
    }

    @Test
    void retainsExplicitTeamScopeAndChecksLiveMembership() {
        long team = insert("INSERT INTO afv_team(name, owner_user_id) VALUES (?, ?)", id(), OWNER.userId());
        jdbc.update("INSERT INTO afv_team_member(team_id, user_id, role) VALUES (?, ?, 1), (?, ?, 3)",
                team, OWNER.userId(), team, OTHER.userId());
        long project = insert("INSERT INTO afv_project(name, owner_type, owner_id) VALUES (?, 2, ?)", id(), team);
        CanvasActor teamOwner = new CanvasActor(OWNER.userId(), team);
        CanvasActor member = new CanvasActor(OTHER.userId(), team);
        String canvas = commands.create(new CanvasCreateCommand(id(), "团队画布", project), teamOwner).id();
        assertThat(queries.snapshot(canvas, member).permissions().canEdit()).isFalse();
        rejects(() -> queries.snapshot(canvas, OTHER), CanvasErrorCode.ACCESS_DENIED);
        jdbc.update("UPDATE afv_team_member SET status = 0 WHERE team_id = ? AND user_id = ?", team, OTHER.userId());
        rejects(() -> queries.events(canvas, member, 0, 10), CanvasErrorCode.ACCESS_DENIED);
    }

    @Test
    void disabledAccountCannotReadCachedContentReplayOrCommit() {
        long user = insert("INSERT INTO sys_user(username, password) VALUES (?, 'test-only')", id());
        CanvasActor creator = new CanvasActor(user, null);
        String canvas = commands.create(new CanvasCreateCommand(id(), "账号状态检查", null), creator).id();
        queries.snapshot(canvas, creator);
        jdbc.update("UPDATE sys_user SET status = 0 WHERE id = ?", user);
        rejects(() -> queries.snapshot(canvas, creator), CanvasErrorCode.ACCESS_DENIED);
        rejects(() -> queries.events(canvas, creator, 0, 10), CanvasErrorCode.ACCESS_DENIED);
        rejects(() -> commands.edit(canvas, command(new CreateNode(text())), creator), CanvasErrorCode.ACCESS_DENIED);
        rejects(() -> queries.list(creator, null, "", false, 1, 10), CanvasErrorCode.ACCESS_DENIED);
    }

    @Test
    void validatesAllBusinessReferenceKindsAndDoesNotPreventMovingDeletedReferences() {
        long project = project(OWNER.userId());
        long script = insert("INSERT INTO afv_script(project_id, title) VALUES (?, ?)", project, id());
        long episode = insert("INSERT INTO afv_script_episode(script_id, episode_number, title) VALUES (?, 1, ?)",
                script, id());
        long scene = insert("INSERT INTO afv_script_scene_item(script_id, episode_id) VALUES (?, ?)", script, episode);
        long storyboard = insert("INSERT INTO afv_storyboard(project_id, script_id, title) VALUES (?, ?, ?)", project,
                script, id());
        long boardEpisode = insert("INSERT INTO afv_storyboard_episode(storyboard_id, episode_number) VALUES (?, 1)",
                storyboard);
        long boardScene = insert("INSERT INTO afv_storyboard_scene(storyboard_id, episode_id) VALUES (?, ?)",
                storyboard, boardEpisode);
        long shot = insert(
                "INSERT INTO afv_storyboard_item(storyboard_id, storyboard_episode_id, storyboard_scene_id) VALUES (?, ?, ?)",
                storyboard, boardEpisode, boardScene);
        long asset = insert("INSERT INTO afv_asset(project_id, name, type) VALUES (?, ?, 'character')", project, id());
        long item = insert("INSERT INTO afv_asset_item(asset_id) VALUES (?)", asset);
        Map<CanvasNodeKind, Long> resources = Map.of(CanvasNodeKind.PROJECT, project, CanvasNodeKind.SCRIPT, script,
                CanvasNodeKind.SCRIPT_EPISODE, episode, CanvasNodeKind.SCRIPT_SCENE, scene,
                CanvasNodeKind.STORYBOARD, storyboard, CanvasNodeKind.STORYBOARD_EPISODE, boardEpisode,
                CanvasNodeKind.STORYBOARD_SCENE, boardScene, CanvasNodeKind.SHOT, shot,
                CanvasNodeKind.ASSET, asset, CanvasNodeKind.ASSET_ITEM, item);
        String canvas = commands.create(new CanvasCreateCommand(id(), "真实业务引用", project), OWNER).id();
        List<CanvasGraphOperation> ops = resources.entrySet().stream().map(entry -> {
            CanvasResourceReference ref = new CanvasResourceReference(CanvasValueType.fromCode(entry.getKey().code()),
                    entry.getValue(), 0L);
            return (CanvasGraphOperation) new CreateNode(
                    new CanvasNode(id(), entry.getKey(), 1, "引用", layout(0, 0, null),
                            mapper.createObjectNode(), ref, 0, 0));
        }).toList();
        commands.edit(canvas, new CanvasEditCommand(id(), ops), OWNER);
        assertThat(queries.snapshot(canvas, OWNER).graph().nodes()).hasSize(10);
        CanvasNode deleted = ((CreateNode) ops.stream()
                .filter(op -> ((CreateNode) op).node().kind() == CanvasNodeKind.SHOT).findFirst().orElseThrow()).node();
        jdbc.update("UPDATE afv_storyboard_item SET deleted = TRUE WHERE id = ?", shot);
        edit(canvas, new LayoutNode(deleted.id(), 0, layout(1, 2, null)));
        CanvasNode rebind = new CanvasNode(id(), deleted.kind(), 1, "删除的镜头", layout(0, 0, null),
                mapper.createObjectNode(), deleted.resource(), 0, 0);
        rejects(() -> edit(canvas, new CreateNode(rebind)), CanvasErrorCode.INVALID_RESOURCE_REFERENCE);
        String otherCanvas = commands.create(new CanvasCreateCommand(id(), "另一个项目", project(OWNER.userId())),
                OWNER).id();
        rejects(() -> commands.edit(otherCanvas, new CanvasEditCommand(id(), ops), OWNER),
                CanvasErrorCode.INVALID_RESOURCE_REFERENCE);
    }

    @Test
    void replaysSnapshotTailAndExplicitlyRejectsExpiredFutureAndGappedCursors() {
        String canvas = create();
        CanvasSnapshot before = queries.snapshot(canvas, OWNER);
        edit(canvas, new CreateNode(text()));
        edit(canvas, new CreateNode(text()));
        assertThat(queries.events(canvas, OWNER, before.lastSequence(), 1).hasMore()).isTrue();
        assertThat(queries.events(canvas, OWNER, 2, 1).events().getFirst().sequence()).isEqualTo(3);
        rejects(() -> queries.events(canvas, OWNER, 4, 10), CanvasErrorCode.INVALID_CURSOR);
        jdbc.update("DELETE FROM afv_canvas_event WHERE canvas_id = ? AND sequence_no = 2", canvas);
        rejects(() -> queries.events(canvas, OWNER, 1, 10), CanvasErrorCode.SNAPSHOT_REQUIRED);
        jdbc.update("DELETE FROM afv_canvas_event WHERE canvas_id = ? AND sequence_no = 1", canvas);
        rejects(() -> queries.events(canvas, OWNER, 0, 10), CanvasErrorCode.SNAPSHOT_REQUIRED);
        assertThat(queries.events(canvas, OWNER, 3, 10).events()).isEmpty();
    }

    @Test
    void liveStreamDeliversCommittedChangesThenTerminatesOnRevocation() {
        String canvas = create();
        share(canvas, OTHER.userId(), false, false);
        CanvasEventStream stream = context.getBean(CanvasEventStream.class);
        StepVerifier.create(stream.open(canvas, OTHER, 1))
                .then(() -> edit(canvas, new CreateNode(text())))
                .assertNext(event -> {
                    assertThat(event.event()).isEqualTo("canvas.graph.changed");
                    assertThat(event.id()).isEqualTo("2");
                })
                .then(() -> jdbc.update("DELETE FROM afv_canvas_member WHERE canvas_id = ?", canvas))
                .assertNext(event -> {
                    assertThat(event.event()).isEqualTo("canvas.error");
                    assertThat(((Map<?, ?>) event.data()).get("errorCode")).isEqualTo(CanvasErrorCode.ACCESS_DENIED);
                })
                .expectComplete().verify(Duration.ofSeconds(8));
    }


    @Test
    void personalCanvasCanReferenceAuthorizedGlobalAssetsWithoutGrantingOthersAssetAccess() {
        String canvas = create();
        long asset = insert(
                "INSERT INTO afv_asset(name, type, owner_type, owner_id, user_id) VALUES ('独立素材', 'prop', 1, ?, ?)",
                OWNER.userId(), OWNER.userId());
        CanvasNode first = new CanvasNode(id(), CanvasNodeKind.ASSET, 1, "素材", layout(0, 0, null),
                mapper.createObjectNode(),
                new CanvasResourceReference(CanvasValueType.ASSET, asset, 0L), 0, 0);
        edit(canvas, new CreateNode(first));
        assertThat(queries.snapshot(canvas, OWNER).graph().nodes()).hasSize(1);
        share(canvas, OTHER.userId(), true, false);
        CanvasNode unauthorized = new CanvasNode(id(), CanvasNodeKind.ASSET, 1, "无权素材", layout(1, 1, null),
                mapper.createObjectNode(),
                new CanvasResourceReference(CanvasValueType.ASSET, asset, 0L), 0, 0);
        rejects(() -> commands.edit(canvas, command(new CreateNode(unauthorized)), OTHER),
                CanvasErrorCode.ACCESS_DENIED);
        long project = project(OWNER.userId());
        long projectAsset = insert(
                "INSERT INTO afv_asset(name, type, project_id, owner_type, owner_id) VALUES ('项目素材', 'prop', ?, 1, ?)",
                project, OWNER.userId());
        CanvasNode wrongScope = new CanvasNode(id(), CanvasNodeKind.ASSET, 1, "项目素材", layout(1, 1, null),
                mapper.createObjectNode(),
                new CanvasResourceReference(CanvasValueType.ASSET, projectAsset, 0L), 0, 0);
        rejects(() -> edit(canvas, new CreateNode(wrongScope)), CanvasErrorCode.INVALID_RESOURCE_REFERENCE);
    }

    @Test
    void rejectedStaleConnectionIntentDoesNotCreateAnOrphanNode() {
        String canvas = create();
        var source = text();
        edit(canvas, new CreateNode(source));
        edit(canvas, new PatchNode(source.id(), 0, "源节点已变化", null));
        var target = node(CanvasNodeKind.COLLECTION, mapper.createObjectNode().put("valueType", "text"));
        long before = count("afv_canvas_operation", canvas);
        rejects(() -> edit(canvas, new CreateNode(target), new CanvasGraphOperation.AssertNode(source.id(), 0),
                new CreateEdge(edge(source.id(), target.id(), 0))), CanvasErrorCode.VERSION_CONFLICT);
        assertThat(queries.snapshot(canvas, OWNER).graph().nodes()).hasSize(1);
        assertThat(count("afv_canvas_operation", canvas)).isEqualTo(before);
        edit(canvas, new CanvasGraphOperation.AssertNode(source.id(), 1), new CreateNode(target),
                new CreateEdge(edge(source.id(), target.id(), 0)));
        assertThat(queries.snapshot(canvas, OWNER).graph().edges()).hasSize(1);
    }

    @Test
    void atomicallyUndoesAndRedoesCreatedNodeAndEdgeWithTheSameIdsAndIncreasingVersions() {
        String canvas = create();
        var source = text();
        var target = node(CanvasNodeKind.COLLECTION, mapper.createObjectNode().put("valueType", "text"));
        edit(canvas, new CreateNode(source));
        var link = edge(source.id(), target.id(), 0);
        var added = edit(canvas, new CreateNode(target), new CreateEdge(link));
        var service = context.getBean(CanvasUndoService.class);
        var request = new CanvasUndoCommand(id(), added.commandId());
        var undone = service.undo(canvas, request, OWNER);
        assertThat(queries.snapshot(canvas, OWNER).graph().nodes()).extracting(CanvasNode::id)
                .containsExactly(source.id());
        assertThat(queries.snapshot(canvas, OWNER).graph().edges()).isEmpty();
        assertThat(service.undo(canvas, request, OWNER)).isEqualTo(undone);
        var redone = service.undo(canvas, new CanvasUndoCommand(id(), undone.commandId()), OWNER);
        var graph = queries.snapshot(canvas, OWNER).graph();
        assertThat(graph.nodes()).extracting(CanvasNode::id).containsExactlyInAnyOrder(source.id(), target.id());
        assertThat(graph.nodes().stream().filter(node -> node.id().equals(target.id())).findFirst().orElseThrow()
                .configVersion()).isEqualTo(2);
        assertThat(graph.edges().getFirst().id()).isEqualTo(link.id());
        assertThat(graph.edges().getFirst().version()).isEqualTo(2);
        service.undo(canvas, new CanvasUndoCommand(id(), redone.commandId()), OWNER);
        assertThat(queries.snapshot(canvas, OWNER).graph().nodes()).hasSize(1);
    }

    @Test
    void conditionalUndoPreservesLaterUnrelatedChangesAndRejectsChangedObjectsAndNewReferences() {
        String canvas = create();
        var first = text();
        var other = text();
        edit(canvas, new CreateNode(first), new CreateNode(other));
        var changed = edit(canvas, new PatchNode(first.id(), 0, "新标题", null));
        edit(canvas, new LayoutNode(other.id(), 0, layout(123, 456, null)));
        var service = context.getBean(CanvasUndoService.class);
        service.undo(canvas, new CanvasUndoCommand(id(), changed.commandId()), OWNER);
        var graph = queries.snapshot(canvas, OWNER).graph();
        assertThat(
                graph.nodes().stream().filter(node -> node.id().equals(other.id())).findFirst().orElseThrow().layout()
                        .x()).isEqualTo(123);
        assertThat(graph.nodes().stream().filter(node -> node.id().equals(first.id())).findFirst().orElseThrow()
                .configVersion()).isEqualTo(2);
        rejects(() -> service.undo(canvas, new CanvasUndoCommand(id(), changed.commandId()), OWNER),
                CanvasErrorCode.VERSION_CONFLICT);
        var target = node(CanvasNodeKind.COLLECTION, mapper.createObjectNode().put("valueType", "text"));
        var added = edit(canvas, new CreateNode(target));
        edit(canvas, new CreateEdge(edge(first.id(), target.id(), 0)));
        rejects(() -> service.undo(canvas, new CanvasUndoCommand(id(), added.commandId()), OWNER),
                CanvasErrorCode.VERSION_CONFLICT);
        assertThat(queries.snapshot(canvas, OWNER).graph().edges()).hasSize(1);
        share(canvas, OTHER.userId(), false, false);
        rejects(() -> service.undo(canvas, new CanvasUndoCommand(id(), added.commandId()), OTHER),
                CanvasErrorCode.ACCESS_DENIED);
        rejects(() -> service.undo(create(), new CanvasUndoCommand(id(), added.commandId()), OWNER),
                CanvasErrorCode.RESOURCE_DELETED);
    }

    @Test
    void undoRestoresDeletedGroupChildrenAndRollsBackWhenJournalFails() {
        String canvas = create();
        var group = node(CanvasNodeKind.GROUP, mapper.createObjectNode());
        var child = new CanvasNode(id(), CanvasNodeKind.TEXT, 1, "子节点", layout(15, 25, group.id()),
                mapper.createObjectNode().put("content", "正文"), null, 0, 0);
        edit(canvas, new CreateNode(group), new CreateNode(child));
        var removed = edit(canvas, new DeleteNode(group.id(), 0, 0));
        var service = context.getBean(CanvasUndoService.class);
        var before = queries.snapshot(canvas, OWNER);
        doThrow(new IllegalStateException("journal unavailable")).when(journal)
                .append(anyLong(), anyString(), isNull(), isNull(),
                        anyString(), any(), any(), any(), anyString());
        assertThatThrownBy(
                () -> service.undo(canvas, new CanvasUndoCommand(id(), removed.commandId()), OWNER)).isInstanceOf(
                IllegalStateException.class);
        reset(journal);
        assertThat(queries.snapshot(canvas, OWNER)).isEqualTo(before);
        service.undo(canvas, new CanvasUndoCommand(id(), removed.commandId()), OWNER);
        assertThat(queries.snapshot(canvas, OWNER).graph().nodes().stream().filter(node -> node.id().equals(child.id()))
                .findFirst().orElseThrow().layout().parentId()).isEqualTo(group.id());
    }

    @Test
    void commentsHistoryMetadataAndCheckpointsUseVersionedAuthorizedTransactions() {
        String canvas = create();
        var node = text();
        edit(canvas, new CreateNode(node));
        var workspace = context.getBean(CanvasWorkspaceService.class);
        String commentId = id();
        var createComment = new CommentCreate(id(), commentId, node.id(), null, "镜头节奏需要放慢");
        var created = workspace.comment(canvas, createComment, OWNER);
        assertThat(workspace.comment(canvas, createComment, OWNER)).isEqualTo(created);
        share(canvas, OTHER.userId(), true, false);
        rejects(() -> workspace.patchComment(canvas, commentId,
                new CommentPatch(id(), 0L, "改写他人的评审", null, null), OTHER), CanvasErrorCode.ACCESS_DENIED);
        workspace.patchComment(canvas, commentId, new CommentPatch(id(), 0L, null, true, null), OTHER);
        rejects(() -> workspace.patchComment(canvas, commentId, new CommentPatch(id(), 0L, null, false, null), OWNER),
                CanvasErrorCode.VERSION_CONFLICT);
        workspace.comment(canvas, new CommentCreate(id(), id(), node.id(), commentId, "已调整"), OTHER);
        assertThat(workspace.comments(canvas, OWNER, 0, 100)).hasSize(2);
        var snapshot = queries.snapshot(canvas, OWNER);
        var checkpoint = new CheckpointCreate(id(), id(), "评审前版本", snapshot.document().revision());
        workspace.checkpoint(canvas, checkpoint, OWNER);
        edit(canvas, new PatchNode(node.id(), 0, "之后的新标题", null));
        assertThat(workspace.checkpoint(canvas, checkpoint.id(), OWNER).snapshot().graph().nodes().getFirst()
                .title()).isEqualTo("节点");
        assertThat(workspace.history(canvas, OWNER, Long.MAX_VALUE, 100)).extracting(
                        CanvasJournalRepository.HistoryEntry::eventType)
                .contains("canvas.comment.created", "canvas.comment.changed", "canvas.checkpoint.created");
        var current = queries.snapshot(canvas, OWNER).document();
        workspace.patch(canvas, new DocumentPatch(id(), current.revision(), "整理后的画布", true), OWNER);
        rejects(() -> edit(canvas, new CreateNode(text())), CanvasErrorCode.CANVAS_ARCHIVED);
        current = queries.snapshot(canvas, OWNER).document();
        workspace.patch(canvas, new DocumentPatch(id(), current.revision(), null, false), OWNER);
        assertThat(queries.snapshot(canvas, OWNER).document().title()).isEqualTo("整理后的画布");
        jdbc.update("DELETE FROM afv_canvas_member WHERE canvas_id = ? AND user_id = ?", canvas, OTHER.userId());
        rejects(() -> workspace.comments(canvas, OTHER, 0, 100), CanvasErrorCode.ACCESS_DENIED);
        rejects(() -> workspace.checkpoint(canvas, checkpoint.id(), OTHER), CanvasErrorCode.ACCESS_DENIED);
    }

    @Test
    void checkpointRetainsDeletedBusinessContentButRechecksCurrentOwnership() {
        String canvas = create();
        long asset = insert(
                "INSERT INTO afv_asset(name, type, description, owner_type, owner_id, user_id) VALUES ('历史素材', 'prop', '保存前的内容', 1, ?, ?)",
                OWNER.userId(), OWNER.userId());
        var node = new CanvasNode(id(), CanvasNodeKind.ASSET, 1, "资产", layout(0, 0, null), mapper.createObjectNode(),
                new CanvasResourceReference(CanvasValueType.ASSET, asset, 0L), 0, 0);
        edit(canvas, new CreateNode(node));
        var workspace = context.getBean(CanvasWorkspaceService.class);
        var checkpoint = new CheckpointCreate(id(), id(), "删除前",
                queries.snapshot(canvas, OWNER).document().revision());
        workspace.checkpoint(canvas, checkpoint, OWNER);
        jdbc.update("UPDATE afv_asset SET deleted = TRUE, version = version + 1 WHERE id = ?", asset);
        rejects(() -> context.getBean(ResourceQueryService.class).get(CanvasValueType.ASSET, asset, OWNER),
                CanvasErrorCode.RESOURCE_DELETED);
        assertThat(workspace.checkpoint(canvas, checkpoint.id(), OWNER).snapshot().resources().getFirst().data()
                .path("description").asText()).isEqualTo("保存前的内容");
        jdbc.update("UPDATE afv_asset SET owner_id = ? WHERE id = ?", OTHER.userId(), asset);
        rejects(() -> workspace.checkpoint(canvas, checkpoint.id(), OWNER), CanvasErrorCode.ACCESS_DENIED);
    }

    private String patchResult(String canvas, String node, String title) {
        try {
            edit(canvas, new PatchNode(node, 1, title, null));
            return "committed";
        } catch (CanvasException failure) {
            return failure.getErrorCode().name();
        }
    }

    private long project(long userId) {
        return insert("INSERT INTO afv_project(name, owner_type, owner_id) VALUES (?, 1, ?)", id(), userId);
    }

    private String create() {
        return commands.create(new CanvasCreateCommand(id(), "画布", null), OWNER).id();
    }

    private CanvasCommandReceipt edit(String canvas, CanvasGraphOperation... ops) {
        return commands.edit(canvas, command(ops), OWNER);
    }

    private CanvasEditCommand command(CanvasGraphOperation... ops) {
        return new CanvasEditCommand(id(), List.of(ops));
    }

    private CanvasNode text() {
        return node(CanvasNodeKind.TEXT, mapper.createObjectNode().put("content", "原文"));
    }

    private CanvasNode node(CanvasNodeKind kind, ObjectNode config) {
        return new CanvasNode(id(), kind, 1, "节点", layout(0, 0, null), config, null, 0, 0);
    }

    private CanvasNodeLayout layout(double x, double y, String parent) {
        return new CanvasNodeLayout(x, y, null, null, parent, false, false);
    }

    private CanvasEdge edge(String source, String target, int index) {
        return new CanvasEdge(id(), CanvasEdgeKind.EXECUTION, source, "text", target, "items", index, null, 0);
    }

    private CanvasEdge at(CanvasEdge edge, int index) {
        return new CanvasEdge(edge.id(), edge.kind(), edge.sourceNodeId(), edge.sourcePort(), edge.targetNodeId(),
                edge.targetPort(), index, null, edge.version());
    }

    private String id() {
        return UUID.randomUUID().toString();
    }

    private long count(String table, String canvas) {
        return Objects.requireNonNull(
                jdbc.queryForObject("SELECT COUNT(*) FROM " + table + " WHERE canvas_id = ?", Long.class, canvas));
    }

    private void share(String canvas, long userId, boolean edit, boolean execute) {
        jdbc.update(
                "INSERT INTO afv_canvas_member(canvas_id, user_id, can_edit, can_execute, granted_by) VALUES (?, ?, ?, ?, ?)",
                canvas, userId, edit, execute, OWNER.userId());
    }

    private void rejects(Runnable operation, CanvasErrorCode code) {
        assertThatThrownBy(operation::run).isInstanceOfSatisfying(CanvasException.class,
                failure -> assertThat(failure.getErrorCode()).isEqualTo(code));
    }

    private long insert(String sql, Object... args) {
        GeneratedKeyHolder key = new GeneratedKeyHolder();
        jdbc.update(connection -> {
            var statement = connection.prepareStatement(sql, Statement.RETURN_GENERATED_KEYS);
            for (int index = 0; index < args.length; index++) {
                statement.setObject(index + 1, args[index]);
            }
            return statement;
        }, key);
        return Objects.requireNonNull(key.getKey()).longValue();
    }
}
