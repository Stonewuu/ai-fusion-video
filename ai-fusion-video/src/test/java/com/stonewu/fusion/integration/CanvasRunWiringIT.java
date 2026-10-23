package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.ai.run.AgentExecutionLimits;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Claim;
import com.stonewu.fusion.service.canvas.run.CanvasRunTransactions;
import com.stonewu.fusion.service.system.SystemConfigService;
import com.stonewu.fusion.config.EpisodeBgmProperties;
import org.springframework.cache.CacheManager;
import com.stonewu.fusion.service.media.job.MediaJobRepository;
import com.stonewu.fusion.service.generation.job.GenerationAttemptRepository;
import com.stonewu.fusion.service.generation.job.GenerationProviderReceipt;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.MediaAccessTickets;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CountDownLatch;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.awaitility.Awaitility.await;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.PROJECT;

/** 真实 HTTP、调度器、MySQL/Redis、托管图片及本地供应商；不调用付费服务。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class CanvasRunWiringIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "canvas-run-wiring-" + UUID.randomUUID()).toAbsolutePath();
    static final AtomicLong IDS = new AtomicLong(38000);
    static final ObjectMapper JSON = new ObjectMapper();
    static final Map<String, AtomicInteger> POSTS = new ConcurrentHashMap<>();
    static final Map<String, AudioScenario> AUDIO = new ConcurrentHashMap<>();
    static final Map<String, TextScenario> TEXT = new ConcurrentHashMap<>();
    static final Map<String, byte[]> IMAGE_EDIT_OUTPUTS = new ConcurrentHashMap<>();
    static final Map<String, byte[]> IMAGE_EDIT_REQUESTS = new ConcurrentHashMap<>();
    static final byte[] PNG = png();
    static final HttpServer PROVIDER = provider();
    @DynamicPropertySource static void properties(DynamicPropertyRegistry p) {
        p.add("spring.datasource.url", MYSQL::getJdbcUrl);
        p.add("spring.datasource.username", MYSQL::getUsername);
        p.add("spring.datasource.password", MYSQL::getPassword);
        p.add("spring.data.redis.host", REDIS::getHost);
        p.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        p.add("spring.data.redis.password", () -> "");
        p.add("app.storage.local-base-path", FILES::toString);
        p.add("app.agent-workspace.local-base-path", () -> FILES.resolve("workspace").toString());
    }
    @Autowired TestRestTemplate http;
    @Autowired TokenService tokens;
    @Autowired JdbcTemplate jdbc;
    @Autowired CanvasRunTransactions runtime;
    @Autowired PlatformTransactionManager manager;
    @Autowired SystemConfigService system;
    @Autowired ResourceCommandService resourceCommands;
    @Autowired ResourceQueryService resourceQueries;
    @Autowired MediaAccessTickets mediaTickets;
    @Autowired EpisodeBgmProperties bgmProperties;
    @Autowired CacheManager caches;
    @Autowired MediaJobRepository mediaJobs;
    @Autowired GenerationAttemptRepository generationAttempts;
    @AfterAll static void closeProvider() { PROVIDER.stop(0); }

    @Test void nodeActivitiesKeepIndependentLatestResultsAndRejectOtherWorkspaces() {
        var f = fixture();
        String first = id(), second = id();
        create(f, List.of(node(first, "text", Map.of("content", "甲")), node(second, "textCombine", Map.of("separator", ""))),
                List.of(edge(first, "text", second, "texts", 0)));
        String firstRun = start(f, plan(f, "SINGLE", List.of(first), Map.of()));
        var firstResult = awaitState(f, firstRun, "SUCCEEDED");
        String secondRun = start(f, plan(f, "DEPENDENCIES", List.of(second), Map.of(first, nodeResult(firstResult, first).path("id").asText())));
        awaitState(f, secondRun, "SUCCEEDED");
        var activities = ok(get(f, "/api/canvases/" + f.canvas + "/node-activities"));
        assertThat(activities).hasSize(2);
        Map<String, String> latest = new HashMap<>();
        activities.forEach(item -> {
            latest.put(item.at("/node/nodeId").asText(), item.path("runId").asText());
            assertThat(item.at("/node/status").asText()).isEqualTo("SUCCEEDED");
            assertThat(item.path("canControl").asBoolean()).isTrue();
        });
        assertThat(latest).containsEntry(first, firstRun).containsEntry(second, secondRun);
        var other = fixture();
        assertThat(get(other, "/api/canvases/" + f.canvas + "/node-activities").getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void activeNodesCannotBeSubmittedTwiceAndAcceptedRequestsRemainIdempotent() {
        var f = fixture();
        String source = id(), choose = id();
        create(f, List.of(node(source, "text", Map.of("content", "候选")), node(choose, "selection", Map.of("valueType", "text"))),
                List.of(edge(source, "text", choose, "candidates", 0)));
        var initial = plan(f, "ALL", List.of(), Map.of());
        var request = startRequest(f, initial);
        String run = ok(post(f, "/api/canvases/" + f.canvas + "/runs", request)).path("id").asText();
        awaitState(f, run, "WAITING_SELECTION");
        assertThat(ok(post(f, "/api/canvases/" + f.canvas + "/runs", request)).path("id").asText()).isEqualTo(run);
        var repeated = plan(f, "ALL", List.of(), Map.of());
        assertThat(post(f, "/api/canvases/" + f.canvas + "/runs", startRequest(f, repeated)).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_run WHERE canvas_id = ?", Integer.class, f.canvas)).isEqualTo(1);
        ok(post(f, "/api/canvas-runs/" + run + "/cancel", Map.of("scope", scope(f))));
    }

    @Test void nodeActivitiesRedactDeletedSourceContent() {
        var f = fixture();
        String media = upload(f, PNG, "inline-private.png", "image/png"), source = id();
        create(f, List.of(node(source, "image", Map.of("resourceId", media))), List.of());
        awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "SUCCEEDED");
        jdbc.update("UPDATE afv_media_resource SET state = 'DELETED', version = version + 1 WHERE id = ?", media);
        var activity = ok(get(f, "/api/canvases/" + f.canvas + "/node-activities")).get(0);
        assertThat(activity.at("/node/contentUnavailable").asBoolean()).isTrue();
        assertThat(activity.at("/node/outputs/values")).isEmpty();
        assertThat(activity.at("/node/inputs").isNull()).isTrue();
        assertThat(activity.toString()).doesNotContain(media);
    }

    @Test void mediaRetentionProjectsLiveNodesCommandsCheckpointsAndFrozenRunsWithoutAnAsyncGap() {
        var f = fixture();
        String media = upload(f, PNG, "retained.png", "image/png");
        String source = id(), choose = id(), note = id(), plainNote = id();
        create(f, List.of(node(source, "image", Map.of("resourceId", media)), node(note, "note", Map.of("content", "说明 media://" + media)),
                node(plainNote, "note", Map.of("content", "没有类型的标识 " + media)),
                node(choose, "selection", Map.of("valueType", "image"))), List.of(edge(source, "media", choose, "candidates", 0)));
        var p = plan(f, "ALL", List.of(), Map.of());
        String run = start(f, p);
        awaitState(f, run, "WAITING_SELECTION");
        String checkpoint = id();
        jdbc.update("""
                INSERT INTO afv_canvas_revision(id, canvas_id, title, revision, sequence_no, snapshot_json, created_by)
                SELECT ?, id, '保留快照', revision, next_event_sequence, JSON_OBJECT('resources', JSON_ARRAY(JSON_OBJECT('data',
                    JSON_OBJECT('videoUrls', JSON_ARRAY(?))))), ? FROM afv_canvas WHERE id = ?
                """, checkpoint, "media://" + media, f.id, f.canvas);
        var kinds = jdbc.queryForList("SELECT DISTINCT anchor_kind FROM afv_media_reference WHERE media_id = ?", String.class, media);
        assertThat(kinds).contains("CANVAS_NODE", "CANVAS_COMMAND", "CANVAS_REVISION", "RUN_PLAN", "CANVAS_RUN", "NODE_RUN");
        // .35 的保留协议包含正文内嵌的规范 URI；只有无类型的裸 UUID 不构成引用。
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE anchor_kind = 'CANVAS_NODE' AND anchor_id = ?", Integer.class, note)).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE anchor_kind = 'CANVAS_NODE' AND anchor_id = ?", Integer.class, plainNote)).isZero();
        commands(f, List.of(Map.of("type", "canvas.node.delete", "nodeId", source, "expectedConfigVersion", 0, "expectedLayoutVersion", 0)));
        jdbc.update("UPDATE afv_canvas_run_plan SET expires_at = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?", p.path("id").asText());
        assertThat(jdbc.queryForObject("SELECT retention_class FROM afv_media_reference WHERE anchor_kind = 'CANVAS_NODE' AND anchor_id = ?", String.class, source)).isEqualTo("HISTORY");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE anchor_kind = 'RUN_PLAN' AND anchor_id = ?", Integer.class, p.path("id").asText())).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE media_id = ? AND anchor_kind = 'CANVAS_RUN' AND anchor_id = ?", Integer.class, media, run)).isEqualTo(1);
        new TransactionTemplate(manager).executeWithoutResult(status -> {
            jdbc.update("UPDATE afv_canvas_node SET config_json = JSON_OBJECT() WHERE id = ?", source);
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE anchor_kind = 'CANVAS_NODE' AND anchor_id = ?", Integer.class, source)).isZero();
            status.setRollbackOnly();
        });
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE anchor_kind = 'CANVAS_NODE' AND anchor_id = ?", Integer.class, source)).isEqualTo(1);
        ok(post(f, "/api/canvas-runs/" + run + "/cancel", Map.of("scope", scope(f))));
    }

    @Test void unavailableMediaIsRedactedWithoutPreventingTheOriginalOwnerFromCancelling() {
        var f = fixture();
        String media = upload(f, PNG, "private-source.png", "image/png");
        String source = id(), choose = id();
        create(f, List.of(node(source, "image", Map.of("resourceId", media)),
                node(choose, "selection", Map.of("valueType", "image"))), List.of(edge(source, "media", choose, "candidates", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        awaitState(f, run, "WAITING_SELECTION");
        jdbc.update("UPDATE afv_media_resource SET state = 'DELETED', version = version + 1 WHERE id = ?", media);
        var view = ok(get(f, "/api/canvas-runs/" + run));
        assertThat(view.path("canControl").asBoolean()).isTrue();
        assertThat(view.toString()).doesNotContain(media, "private-source.png");
        assertThat(nodeResult(view, choose).path("contentUnavailable").asBoolean()).isTrue();
        assertThat(nodeResult(view, choose).path("inputs").isNull()).isTrue();
        assertThat(view.at("/statistics/media/unavailableNodes").asInt()).isEqualTo(2);
        ok(post(f, "/api/canvas-runs/" + run + "/cancel", Map.of("scope", scope(f))));
        assertThat(awaitState(f, run, "CANCELLED").toString()).doesNotContain(media);
        var other = fixture();
        assertThat(get(other, "/api/canvas-runs/" + run).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void deletedBusinessSourceDoesNotTurnAReadableHistoricalRunIntoARollbackError() {
        var f = fixture();
        long asset = IDS.incrementAndGet();
        jdbc.update("INSERT INTO afv_asset(id, name, type, project_id, user_id, owner_type, owner_id) VALUES (?, '已删除角色', 'character', ?, ?, 1, ?)", asset, f.id, f.id, f.id);
        String source = id();
        create(f, List.of(resourceNode(source, "asset", asset)), List.of());
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        awaitState(f, run, "SUCCEEDED");
        jdbc.update("UPDATE afv_asset SET deleted = TRUE, version = version + 1 WHERE id = ?", asset);
        var view = ok(get(f, "/api/canvas-runs/" + run));
        assertThat(view.path("status").asText()).isEqualTo("SUCCEEDED");
        assertThat(nodeResult(view, source).path("derivedState").asText()).isEqualTo("SOURCE_UNAVAILABLE");
    }

    @Test void importedHistoricalSourceFeedsItsOriginalBusinessPortWithoutDownloadingOrRewritingTheBusiness() throws Exception {
        var f = fixture();
        long asset = IDS.incrementAndGet(), item = IDS.incrementAndGet();
        String filename = "imported-source-" + item + ".png";
        Files.createDirectories(FILES);
        Files.write(FILES.resolve(filename), PNG);
        jdbc.update("INSERT INTO afv_asset(id, name, type, project_id, user_id, owner_type, owner_id) VALUES (?, '来源角色', 'character', ?, ?, 1, ?)", asset, f.id, f.id, f.id);
        jdbc.update("INSERT INTO afv_asset_item(id, asset_id, name, image_url) VALUES (?, ?, '参考图', ?)", item, asset, "/media/" + filename);
        String source = id(), collection = id();
        create(f, List.of(resourceNode(source, "assetItem", item), node(collection, "collection", Map.of("valueType", "image"))),
                List.of(edge(source, "image", collection, "items", 0)));
        assertThat(plan(f, "ALL", List.of(), Map.of()).path("issues")).isNotEmpty();
        String imported = ok(post(f, "/api/canvases/" + f.canvas + "/media-imports", Map.of("requestId", id(), "scope", scope(f),
                "sources", List.of(Map.of("title", "参考图", "source", Map.of("type", "assetItem", "id", item, "version", 0, "field", "imageUrl")))))).path("id").asText();
        await().pollInterval(Duration.ofMillis(200)).atMost(Duration.ofSeconds(20)).untilAsserted(() ->
                assertThat(ok(get(f, "/api/media-imports/" + imported)).path("status").asText()).isEqualTo("SUCCEEDED"));
        String mediaId = ok(get(f, "/api/media-imports/" + imported)).at("/items/0/resource/id").asText();
        Files.delete(FILES.resolve(filename));
        var run = awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "SUCCEEDED");
        assertThat(nodeResult(run, source).at("/outputs/values/image/0/mediaId").asText()).isEqualTo(mediaId);
        assertThat(nodeResult(run, collection).at("/outputs/values/items/0/mediaId").asText()).isEqualTo(mediaId);
        assertThat(jdbc.queryForObject("SELECT image_url FROM afv_asset_item WHERE id = ?", String.class, item)).isEqualTo("/media/" + filename);
        jdbc.update("UPDATE afv_asset_item SET image_url = '/media/changed-without-version.png' WHERE id = ?", item);
        assertThat(plan(f, "ALL", List.of(), Map.of()).path("issues")).isNotEmpty();
    }

    @Test void writebackWaitsForBoundApprovalAndAtomicallyPublishesOneUpdatedResource() {
        var f = fixture();
        String source = id(), target = id(), write = id(), next = id();
        create(f, List.of(node(source, "text", Map.of("content", "确认后写入的项目介绍")), resourceNode(target, "project", f.id),
                node(write, "writeback", writeback("project", "text", "description")), node(next, "collection", Map.of("valueType", "project"))),
                List.of(edge(source, "text", write, "value", 0), edge(target, "project", write, "target", 0), edge(write, "resource", next, "items", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var pending = nodeResult(awaitState(f, run, "WAITING_CONFIRMATION"), write);
        assertThat(jdbc.queryForObject("SELECT description FROM afv_project WHERE id = ?", String.class, f.id)).isNull();
        assertThat(pending.at("/writeback/value/text").asText()).isEqualTo("确认后写入的项目介绍");
        String path = "/api/canvas-runs/" + run + "/nodes/" + pending.path("id").asText() + "/writeback";
        var decision = decision(f, pending, "APPROVE");
        var wrongHash = new HashMap<>(decision); wrongHash.put("proposalHash", "not-the-preview");
        assertThat(post(f, path, wrongHash).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        ok(post(f, path, decision)); ok(post(f, path, decision));
        var done = awaitState(f, run, "SUCCEEDED");
        assertThat(jdbc.queryForObject("SELECT description FROM afv_project WHERE id = ?", String.class, f.id)).isEqualTo("确认后写入的项目介绍");
        assertThat(jdbc.queryForObject("SELECT version FROM afv_project WHERE id = ?", Integer.class, f.id)).isEqualTo(1);
        assertThat(nodeResult(done, next).at("/outputs/values/items/0/resource/version").asInt()).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE request_id = ?", Integer.class,
                "canvas-writeback:" + pending.path("id").asText())).isEqualTo(1);
        assertThat(nodeResult(done, write).at("/writeback/decision/commandId").asText()).isNotBlank();
    }

    @Test void writebackDoesNotOverwriteConcurrentResourceEditsAndCanBeSkipped() {
        var f = fixture(); String source = id(), target = id(), write = id(), next = id();
        create(f, List.of(node(source, "text", Map.of("content", "旧计划的生成结果")), resourceNode(target, "project", f.id),
                node(write, "writeback", writeback("project", "text", "description")), node(next, "collection", Map.of("valueType", "project"))),
                List.of(edge(source, "text", write, "value", 0), edge(target, "project", write, "target", 0), edge(write, "resource", next, "items", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var pending = nodeResult(awaitState(f, run, "WAITING_CONFIRMATION"), write);
        resourceCommands.execute(new ResourcePatchCommand(id(), PROJECT, f.id, 0, JSON.valueToTree(Map.of("description", "用户在原页面的新编辑"))),
                ResourceCommandContext.user(new CanvasActor(f.id, null), f.canvas));
        String path = "/api/canvas-runs/" + run + "/nodes/" + pending.path("id").asText() + "/writeback";
        assertThat(post(f, path, decision(f, pending, "APPROVE")).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(jdbc.queryForObject("SELECT description FROM afv_project WHERE id = ?", String.class, f.id)).isEqualTo("用户在原页面的新编辑");
        ok(post(f, path, decision(f, pending, "SKIP")));
        var done = awaitState(f, run, "PARTIAL_FAILED");
        assertThat(nodeResult(done, next).path("status").asText()).isEqualTo("BLOCKED");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE request_id = ?", Integer.class,
                "canvas-writeback:" + pending.path("id").asText())).isZero();
    }

    @Test void writebackRejectsAnotherActorAndApprovalAfterCancellation() {
        var f = fixture(); var other = fixture(); String source = id(), target = id(), write = id();
        create(f, List.of(node(source, "text", Map.of("content", "不应写入")), resourceNode(target, "project", f.id),
                node(write, "writeback", writeback("project", "text", "description"))),
                List.of(edge(source, "text", write, "value", 0), edge(target, "project", write, "target", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var pending = nodeResult(awaitState(f, run, "WAITING_CONFIRMATION"), write);
        String path = "/api/canvas-runs/" + run + "/nodes/" + pending.path("id").asText() + "/writeback";
        assertThat(post(other, path, decision(other, pending, "APPROVE")).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        ok(post(f, "/api/canvas-runs/" + run + "/cancel", Map.of("scope", scope(f))));
        awaitState(f, run, "CANCELLED");
        assertThat(post(f, path, decision(f, pending, "APPROVE")).getStatusCode().is4xxClientError()).isTrue();
        assertThat(jdbc.queryForObject("SELECT description FROM afv_project WHERE id = ?", String.class, f.id)).isNull();
    }

    @Test void writebackBindsManagedImageWithParentAssetProtection() {
        var f = fixture();
        jdbc.update("INSERT INTO afv_asset(id, project_id, name, type, owner_type, owner_id) VALUES (?, ?, '写入测试资产', 'character', 1, ?)", f.id, f.id, f.id);
        jdbc.update("INSERT INTO afv_asset_item(id, asset_id, name, item_type) VALUES (?, ?, '原始形象', 'original')", f.id, f.id);
        String media = upload(f, PNG, "writeback.png", "image/png");
        String source = id(), target = id(), write = id();
        create(f, List.of(node(source, "image", Map.of("resourceId", media)), resourceNode(target, "assetItem", f.id),
                node(write, "writeback", writeback("assetItem", "image", "imageUrl"))),
                List.of(edge(source, "media", write, "value", 0), edge(target, "assetItem", write, "target", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var pending = nodeResult(awaitState(f, run, "WAITING_CONFIRMATION"), write);
        assertThat(pending.at("/writeback/expectedParentVersion").asInt()).isZero();
        ok(post(f, "/api/canvas-runs/" + run + "/nodes/" + pending.path("id").asText() + "/writeback", decision(f, pending, "APPROVE")));
        awaitState(f, run, "SUCCEEDED");
        assertThat(jdbc.queryForObject("SELECT image_url FROM afv_asset_item WHERE id = ?", String.class, f.id)).isEqualTo("media://" + media);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_asset WHERE id = ?", Integer.class, f.id)).isEqualTo(1);
    }

    @Test void writebackShowsScriptResetImpactAndRequiresExplicitResetAcknowledgement() {
        var f = fixture();
        jdbc.update("INSERT INTO afv_script(id, project_id, title, raw_content) VALUES (?, ?, '原剧本', '原有原文')", f.id, f.id);
        jdbc.update("INSERT INTO afv_script_episode(id, script_id, episode_number, title) VALUES (?, ?, 1, '将被重置的一集')", f.id, f.id);
        jdbc.update("INSERT INTO afv_script_scene_item(id, script_id, episode_id, scene_number, scene_heading) VALUES (?, ?, ?, '1', '将被重置的一场')", f.id, f.id, f.id);
        String source = id(), target = id(), write = id();
        create(f, List.of(node(source, "text", Map.of("content", "确认后的新剧本原文")), resourceNode(target, "script", f.id),
                node(write, "writeback", writeback("script", "text", "rawContent"))),
                List.of(edge(source, "text", write, "value", 0), edge(target, "script", write, "target", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var pending = nodeResult(awaitState(f, run, "WAITING_CONFIRMATION"), write);
        assertThat(pending.at("/writeback/dangerous").asBoolean()).isTrue();
        assertThat(pending.at("/writeback/removedResources")).hasSize(2);
        String path = "/api/canvas-runs/" + run + "/nodes/" + pending.path("id").asText() + "/writeback";
        var payload = new HashMap<>(decision(f, pending, "APPROVE"));
        assertThat(post(f, path, payload).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_script_episode WHERE id = ?", Boolean.class, f.id)).isFalse();
        payload.put("acceptReset", true); ok(post(f, path, payload));
        awaitState(f, run, "SUCCEEDED");
        assertThat(jdbc.queryForObject("SELECT raw_content FROM afv_script WHERE id = ?", String.class, f.id)).isEqualTo("确认后的新剧本原文");
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_script_episode WHERE id = ?", Boolean.class, f.id)).isTrue();
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_script_scene_item WHERE id = ?", Boolean.class, f.id)).isTrue();
    }

    @Test void reparsedScriptRequiresApprovalKeepsOldReferencesExplicitAndAllowsRebindingWithoutLosingMedia() throws Exception {
        var f = fixture();
        // 本用例通过业务命令生成子 ID，给后续显式 ID 夹具保留独立区间。
        IDS.addAndGet(10_000);
        jdbc.update("INSERT INTO afv_script(id,project_id,title,raw_content) VALUES(?,?,'重解析验收','旧原文')", f.id, f.id);
        jdbc.update("INSERT INTO afv_script_episode(id,script_id,episode_number,title) VALUES(?,?,1,'原分集')", f.id, f.id);
        jdbc.update("INSERT INTO afv_script_scene_item(id,script_id,episode_id,scene_number,scene_heading) VALUES(?,?,?,'1','原场次')", f.id, f.id, f.id);
        String media = upload(f, PNG, "已生成的首帧.png", "image/png");
        String text = id(), script = id(), write = id(), oldEpisode = id(), oldScene = id(), image = id();
        create(f, List.of(node(text, "text", Map.of("content", "雨夜，阿青在灯塔前读信。")), resourceNode(script, "script", f.id),
                        node(write, "writeback", writeback("script", "text", "rawContent")), resourceNode(oldEpisode, "scriptEpisode", f.id),
                        resourceNode(oldScene, "scriptScene", f.id), node(image, "image", Map.of("resourceId", media))),
                List.of(edge(text, "text", write, "value", 0), edge(script, "script", write, "target", 0),
                        Map.of("id", id(), "kind", "annotation", "sourceNodeId", oldScene, "targetNodeId", image, "version", 0)));
        String run = start(f, plan(f, "DEPENDENCIES", List.of(write), Map.of()));
        var pending = nodeResult(awaitState(f, run, "WAITING_CONFIRMATION"), write);
        assertThat(pending.at("/writeback/dangerous").asBoolean()).isTrue();
        assertThat(pending.at("/writeback/removedResources")).hasSize(2);
        String path = "/api/canvas-runs/" + run + "/nodes/" + pending.path("id").asText() + "/writeback";
        var approval = new HashMap<>(decision(f, pending, "APPROVE"));
        assertThat(post(f, path, approval).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_script_scene_item WHERE id=?", Boolean.class, f.id)).isFalse();
        approval.put("acceptReset", true);
        ok(post(f, path, approval));
        awaitState(f, run, "SUCCEEDED");
        var actor = new CanvasActor(f.id, null);
        assertThatThrownBy(() -> resourceQueries.get(CanvasValueType.SCRIPT_SCENE, f.id, actor))
                .isInstanceOf(CanvasException.class).hasMessageContaining("已删除");
        Map<String, JsonNode> beforeRebind = new HashMap<>();
        snapshot(f).at("/graph/nodes").forEach(value -> beforeRebind.put(value.path("id").asText(), value));
        assertThat(beforeRebind.get(oldScene).at("/resource/id").asLong()).isEqualTo(f.id);
        assertThat(beforeRebind.get(oldEpisode).at("/resource/id").asLong()).isEqualTo(f.id);

        var context = ResourceCommandContext.user(actor, f.canvas);
        var episode = resourceCommands.execute(new ResourceCreateCommand(id(), CanvasValueType.SCRIPT_EPISODE, f.id,
                JSON.createObjectNode().put("scriptId", f.id).put("parentVersion", 1).put("episodeNumber", 1).put("title", "重解析后的新分集")), context);
        long episodeId = episode.result().path("id").asLong();
        var scene = resourceCommands.execute(new ResourceCreateCommand(id(), CanvasValueType.SCRIPT_SCENE, f.id,
                JSON.createObjectNode().put("episodeId", episodeId).put("parentVersion", 0).put("sceneNumber", "1").put("sceneHeading", "重解析后的新场次")), context);
        long sceneId = scene.result().path("id").asLong();
        assertThat(sceneId).isNotEqualTo(f.id);
        commands(f, List.of(Map.of("type", "canvas.node.bind", "nodeId", oldEpisode, "expectedConfigVersion", 0,
                        "resource", Map.of("type", "scriptEpisode", "id", episodeId, "version", 1)),
                Map.of("type", "canvas.node.bind", "nodeId", oldScene, "expectedConfigVersion", 0,
                        "resource", Map.of("type", "scriptScene", "id", sceneId, "version", 0))));
        Map<String, JsonNode> afterRebind = new HashMap<>();
        var restored = snapshot(f);
        restored.at("/graph/nodes").forEach(value -> afterRebind.put(value.path("id").asText(), value));
        assertThat(afterRebind.get(oldEpisode).at("/resource/id").asLong()).isEqualTo(episodeId);
        assertThat(afterRebind.get(oldScene).at("/resource/id").asLong()).isEqualTo(sceneId);
        assertThat(afterRebind.get(image).at("/config/resourceId").asText()).isEqualTo(media);
        assertThat(restored.at("/graph/nodes")).hasSize(6);
        assertThat(restored.at("/graph/edges")).hasSize(3);
        var ticket = mediaTickets.browser(media, actor, false);
        assertThat(ticket.url()).startsWith("/api/media-content/");
        var download = http.exchange(ticket.url(), HttpMethod.GET, new HttpEntity<>(f.headers), byte[].class);
        assertThat(download.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(download.getBody()).isEqualTo(PNG);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission WHERE project_id=?", Integer.class, f.id)).isZero();
        Files.writeString(Path.of("target/g-script-reparse-journey-results.json"), JSON.writerWithDefaultPrettyPrinter().writeValueAsString(Map.of(
                "resetImpactCount", 2, "approvalWithoutResetAcknowledgementRejected", true, "oldReferencesRemainExplicit", true,
                "newEpisodeId", episodeId, "newSceneId", sceneId, "reboundNodeCount", 2, "graphNodes", 6, "graphEdges", 3,
                "originalMediaDownloadMatches", true, "generationSubmissions", 0)));
    }

    @Test void writebackAdoptsEpisodeMediaWithoutRequiringExistingShotVideos() throws Exception {
        var f = fixture();
        jdbc.update("INSERT INTO afv_storyboard(id, project_id, title) VALUES (?, ?, '分镜')", f.id, f.id);
        jdbc.update("INSERT INTO afv_storyboard_episode(id, storyboard_id, episode_number, title) VALUES (?, ?, 1, '还没有镜头视频')", f.id, f.id);
        String media = upload(f, video(f, "blue", false), "new-composition.mp4", "video/mp4");
        String source = id(), target = id(), write = id();
        create(f, List.of(node(source, "video", Map.of("resourceId", media)), resourceNode(target, "storyboardEpisode", f.id),
                node(write, "writeback", writeback("storyboardEpisode", "video", "composedVideoUrl"))),
                List.of(edge(source, "media", write, "value", 0), edge(target, "episode", write, "target", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var pending = nodeResult(awaitState(f, run, "WAITING_CONFIRMATION"), write);
        ok(post(f, "/api/canvas-runs/" + run + "/nodes/" + pending.path("id").asText() + "/writeback", decision(f, pending, "APPROVE")));
        awaitState(f, run, "SUCCEEDED");
        assertThat(jdbc.queryForObject("SELECT composed_video_url FROM afv_storyboard_episode WHERE id = ?", String.class, f.id)).isEqualTo("media://" + media);
        assertThat(jdbc.queryForObject("SELECT compose_status FROM afv_storyboard_episode WHERE id = ?", Integer.class, f.id)).isEqualTo(2);
        assertThat(jdbc.queryForObject("SELECT compose_job_id FROM afv_storyboard_episode WHERE id = ?", String.class, f.id)).isNull();
    }

    @Test void writebackKeepsFrozenEpisodeClipsForAnExplicitDownstreamComposition() throws Exception {
        var f = fixture();
        String media = upload(f, video(f, "green", false), "shot-clip.mp4", "video/mp4");
        String generated = upload(f, video(f, "red", false), "unselected-generation.mp4", "video/mp4");
        jdbc.update("INSERT INTO afv_storyboard(id, project_id, title) VALUES (?, ?, '可合成分镜')", f.id, f.id);
        jdbc.update("INSERT INTO afv_storyboard_episode(id, storyboard_id, episode_number, title) VALUES (?, ?, 1, '带镜头视频')", f.id, f.id);
        jdbc.update("INSERT INTO afv_storyboard_scene(id, storyboard_id, episode_id) VALUES (?, ?, ?)", f.id, f.id, f.id);
        jdbc.update("INSERT INTO afv_storyboard_item(id, storyboard_id, storyboard_episode_id, storyboard_scene_id, video_url, generated_video_url) VALUES (?, ?, ?, ?, ?, ?)",
                f.id, f.id, f.id, f.id, "media://" + media, "media://" + generated);
        String source = id(), target = id(), write = id(), compose = id(), shot = id(), selectedVideo = id();
        create(f, List.of(node(source, "text", Map.of("content", "更新分集说明后合成")), resourceNode(target, "storyboardEpisode", f.id),
                node(write, "writeback", writeback("storyboardEpisode", "text", "synopsis")), node(compose, "videoCompose", Map.of("resolution", "128x96", "frameRate", 24)),
                resourceNode(shot, "shot", f.id), node(selectedVideo, "collection", Map.of("valueType", "video"))),
                List.of(edge(source, "text", write, "value", 0), edge(target, "episode", write, "target", 0), edge(write, "resource", compose, "episode", 0),
                        edge(shot, "video", selectedVideo, "items", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var pending = nodeResult(awaitState(f, run, "WAITING_CONFIRMATION"), write);
        ok(post(f, "/api/canvas-runs/" + run + "/nodes/" + pending.path("id").asText() + "/writeback", decision(f, pending, "APPROVE")));
        var done = awaitState(f, run, "SUCCEEDED");
        assertThat(nodeResult(done, compose).at("/outputs/values/video/0/mediaId").asText()).isNotBlank();
        assertThat(nodeResult(done, compose).at("/inputs/values/episode/0/data/_canvasVideos/0/mediaId").asText()).isEqualTo(media);
        assertThat(nodeResult(done, selectedVideo).at("/outputs/values/items/0/mediaId").asText()).isEqualTo(media);
    }

    @Test void writebackDoesNotAdoptEpisodeMediaWhileOriginalCompositionIsActive() throws Exception {
        var f = fixture();
        jdbc.update("INSERT INTO afv_storyboard(id, project_id) VALUES (?, ?)", f.id, f.id);
        jdbc.update("INSERT INTO afv_storyboard_episode(id, storyboard_id, episode_number, compose_status, compose_job_id) VALUES (?, ?, 1, 1, ?)", f.id, f.id, id());
        String media = upload(f, video(f, "red", false), "wait-original-compose.mp4", "video/mp4");
        String source = id(), target = id(), write = id();
        create(f, List.of(node(source, "video", Map.of("resourceId", media)), resourceNode(target, "storyboardEpisode", f.id),
                node(write, "writeback", writeback("storyboardEpisode", "video", "composedVideoUrl"))),
                List.of(edge(source, "media", write, "value", 0), edge(target, "episode", write, "target", 0)));
        var done = awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "PARTIAL_FAILED");
        assertThat(nodeResult(done, write).path("error").asText()).contains("仍在合成");
        assertThat(jdbc.queryForObject("SELECT composed_video_url FROM afv_storyboard_episode WHERE id = ?", String.class, f.id)).isNull();
    }

    @Test void batchFreezesTargetsAndPublishesResultsInInputOrder() {
        var f = fixture();
        String a = id(), b = id(), batch = id(), next = id();
        create(f, List.of(node(a, "text", Map.of("content", "甲")), node(b, "text", Map.of("content", "乙")),
                node(batch, "foreach", batch("textCombine", Map.of(), "texts", "text", "text", "text", 2)),
                node(next, "textCombine", Map.of("separator", " / "))),
                List.of(edge(a, "text", batch, "items", 1), edge(b, "text", batch, "items", 0), edge(batch, "results", next, "texts", 0)));
        var plan = plan(f, "ALL", List.of(), Map.of());
        assertThat(plan.at("/cost/batchItems").asInt()).isEqualTo(2);
        assertThat(plan.at("/cost/nodes").asInt()).isEqualTo(6);
        String run = start(f, plan);
        var done = awaitState(f, run, "SUCCEEDED");
        var parent = nodeResult(done, batch);
        var children = children(done, parent.path("id").asText());
        assertThat(children).hasSize(2);
        assertThat(done.at("/statistics/nodes/total").asInt()).isEqualTo(4);
        assertThat(done.at("/statistics/nodes/succeeded").asInt()).isEqualTo(4);
        assertThat(done.at("/statistics/nodes/batchAttempts").asInt()).isEqualTo(2);
        assertThat(children.getFirst().path("itemIndex").asInt()).isZero();
        assertThat(children.getFirst().at("/inputs/values/texts/0/text").asText()).isEqualTo("乙");
        assertThat(nodeResult(done, next).at("/outputs/values/text/0/text").asText()).isEqualTo("乙 / 甲");
        assertThat(jdbc.queryForObject("SELECT JSON_LENGTH(checkpoint_json, '$.targetIds') FROM afv_canvas_node_run WHERE id = ?",
                Integer.class, parent.path("id").asText())).isEqualTo(2);
        assertThat(done.toString()).doesNotContain("apiHash", "plan_json");
        var childReuse = plan(f, "SINGLE", List.of(batch), Map.of(batch, children.getFirst().path("id").asText()));
        assertThat(childReuse.path("issues").toString()).contains("汇总结果");
    }

    @Test void batchRetainsFrozenInputsAcrossLeaseRecoveryAndSourceEdits() throws Exception {
        var f = fixture();
        var scenario = textModel(f, 200, true, "stop");
        String a = id(), b = id(), batch = id();
        create(f, List.of(node(a, "text", Map.of("content", "text-" + f.id + " 原始甲")),
                node(b, "text", Map.of("content", "text-" + f.id + " 原始乙")),
                node(batch, "foreach", batch("textGeneration", Map.of("modelId", f.id), "prompt", "text", "text", "text", 2))),
                List.of(edge(a, "text", batch, "items", 0), edge(b, "text", batch, "items", 1)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        try {
            assertThat(scenario.entered.await(15, TimeUnit.SECONDS)).isTrue();
            String parent = nodeResult(ok(get(f, "/api/canvas-runs/" + run)), batch).path("id").asText();
            jdbc.update("UPDATE afv_canvas_node_run SET owner_instance_id = 'lost-worker', lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?", parent);
            commands(f, List.of(Map.of("type", "canvas.node.patch", "nodeId", b, "expectedConfigVersion", 0,
                    "configPatch", Map.of("content", "编辑后的新输入，不属于本次执行"))));
            scenario.release.countDown();
            var done = awaitState(f, run, "SUCCEEDED");
            assertThat(children(done, parent)).hasSize(2);
            assertThat(children(done, parent).getLast().at("/inputs/values/prompt/0/text").asText()).endsWith("原始乙");
            assertThat(scenario.posts).hasValue(2);
            assertThat(nodeResult(done, batch).path("derivedState").asText()).isEqualTo("STALE");
        } finally { scenario.release.countDown(); }
    }

    @Test void batchRetriesOnlyFailedItemsAfterExplicitReviewWithoutRepeatingSuccesses() {
        var f = fixture();
        textModel(f, 200, false, "stop");
        TEXT.remove("text-" + f.id);
        var good = new TextScenario(200, false, "stop");
        var bad = new TextScenario(503, false, "stop");
        TEXT.put("text-" + f.id + "-good", good); TEXT.put("text-" + f.id + "-bad", bad);
        String a = id(), b = id(), batch = id(), next = id();
        create(f, List.of(node(a, "text", Map.of("content", "text-" + f.id + "-good")),
                node(b, "text", Map.of("content", "text-" + f.id + "-bad")),
                node(batch, "foreach", batch("textGeneration", Map.of("modelId", f.id), "prompt", "text", "text", "text", 2)),
                node(next, "textCombine", Map.of("separator", " / "))),
                List.of(edge(a, "text", batch, "items", 0), edge(b, "text", batch, "items", 1), edge(batch, "results", next, "texts", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var waiting = awaitState(f, run, "WAITING_CONFIRMATION");
        String parent = nodeResult(waiting, batch).path("id").asText();
        assertThat(children(waiting, parent)).anySatisfy(child -> {
            assertThat(child.path("status").asText()).isEqualTo("FAILED");
            assertThat(child.path("error").asText()).contains("HTTP 503", "text queue is full");
        });
        assertThat(nodeResult(waiting, next).path("status").asText()).isEqualTo("WAITING_INPUT");
        var payload = Map.of("expectedRevision", nodeResult(waiting, batch).path("revision").asLong(), "action", "RETRY_FAILED", "note", "只重试已确认失败的项目", "scope", scope(f));
        bad.status = 200;
        String path = "/api/canvas-runs/" + run + "/nodes/" + parent + "/batch/review";
        ok(post(f, path, payload));
        assertThat(post(f, path, payload).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        var done = awaitState(f, run, "SUCCEEDED");
        assertThat(children(done, parent)).hasSize(3);
        assertThat(good.posts).hasValue(1); assertThat(bad.posts).hasValue(2);
        assertThat(children(done, parent).getLast().path("attempt").asInt()).isEqualTo(2);
        assertThat(nodeResult(done, next).at("/outputs/values/text/0/text").asText()).isEqualTo("模型返回文本 / 模型返回文本");
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM afv_canvas_run_event WHERE run_id = ? AND event_type = 'CANVAS_RUN_BATCH_REVIEW'", Integer.class, run)).isEqualTo(1));
    }

    @Test void batchCancellationWaitsForSubmittedItemAndDoesNotStartRemainingItems() throws Exception {
        var f = fixture();
        var scenario = textModel(f, 200, true, "stop");
        String a = id(), b = id(), batch = id();
        create(f, List.of(node(a, "text", Map.of("content", "text-" + f.id + " 第一项")),
                node(b, "text", Map.of("content", "text-" + f.id + " 第二项")),
                node(batch, "foreach", batch("textGeneration", Map.of("modelId", f.id), "prompt", "text", "text", "text", 2))),
                List.of(edge(a, "text", batch, "items", 0), edge(b, "text", batch, "items", 1)));
        var request = Map.of("expectedRevision", snapshot(f).at("/document/revision").asLong(), "mode", "ALL", "nodeIds", List.of(),
                "reuse", Map.of(), "scope", scope(f), "budget", Map.of("maxNodes", 10, "maxImages", 0, "maxVideoSeconds", 0, "maxParallel", 1, "maxBatchItems", 2));
        String run = start(f, ok(post(f, "/api/canvases/" + f.canvas + "/run-plans", request)));
        try {
            assertThat(scenario.entered.await(15, TimeUnit.SECONDS)).isTrue();
            ok(post(f, "/api/canvas-runs/" + run + "/cancel", Map.of("scope", scope(f))));
            await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                    "SELECT COUNT(*) FROM afv_canvas_text_job WHERE run_id = ? AND cancel_requested = TRUE", Integer.class, run)).isEqualTo(1));
            assertThat(ok(get(f, "/api/canvas-runs/" + run)).path("status").asText()).isNotEqualTo("CANCELLED");
            scenario.release.countDown();
            var done = awaitState(f, run, "CANCELLED");
            assertThat(scenario.posts).hasValue(1);
            assertThat(nodeResult(done, batch).path("outputs").path("values")).isEmpty();
            assertThat(children(done, nodeResult(done, batch).path("id").asText())).allSatisfy(child ->
                    assertThat(child.path("status").asText()).isEqualTo("CANCELLED"));
        } finally { scenario.release.countDown(); }
    }

    @Test void batchChecksWorstCaseBudgetBeforeSubmittingAndActualCountBeforeExpansion() {
        var f = fixture(); model(f);
        String a = id(), b = id(), batch = id();
        create(f, List.of(node(a, "text", Map.of("content", "第一项")), node(b, "text", Map.of("content", "第二项")),
                node(batch, "foreach", batch("imageGeneration", Map.of("modelId", f.id), "prompt", "images", "text", "image", 33))),
                List.of(edge(a, "text", batch, "items", 0), edge(b, "text", batch, "items", 1)));
        var excessive = plan(f, "ALL", List.of(), Map.of());
        assertThat(excessive.at("/cost/images").asInt()).isEqualTo(33);
        assertThat(excessive.path("issues").toString()).contains("BUDGET_EXCEEDED");
        assertThat(post(f, "/api/canvases/" + f.canvas + "/runs", startRequest(f, excessive)).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        commands(f, List.of(Map.of("type", "canvas.node.patch", "nodeId", batch, "expectedConfigVersion", 0, "configPatch", Map.of("maxItems", 1))));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var failed = awaitState(f, run, "PARTIAL_FAILED");
        assertThat(nodeResult(failed, batch).path("error").asText()).contains("批量数量");
        assertThat(children(failed, nodeResult(failed, batch).path("id").asText())).isEmpty();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission WHERE canvas_node_run_id IN (SELECT id FROM afv_canvas_node_run WHERE run_id = ?)", Integer.class, run)).isZero();
    }

    @ParameterizedTest @ValueSource(strings = {"USE_SUCCEEDED", "FINISH"})
    void batchRequiresExplicitDecisionForPartialMediaResultsAndEnforcesRetryBudget(String action) throws Exception {
        var f = fixture();
        String normal = upload(f, PNG, "normal.png", "image/png");
        var bytes = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB), "png", bytes);
        String small = upload(f, bytes.toByteArray(), "too-small.png", "image/png");
        String a = id(), b = id(), batch = id(), next = id();
        create(f, List.of(node(a, "image", Map.of("resourceId", normal)), node(b, "image", Map.of("resourceId", small)),
                node(batch, "foreach", batch("imageTransform", Map.of("operation", "crop", "x", 0, "y", 0, "width", 16, "height", 16),
                        "image", "image", "image", "image", 2)), node(next, "collection", Map.of("valueType", "image"))),
                List.of(edge(a, "media", batch, "items", 0), edge(b, "media", batch, "items", 1), edge(batch, "results", next, "items", 0)));
        var request = Map.of("expectedRevision", snapshot(f).at("/document/revision").asLong(), "mode", "ALL", "nodeIds", List.of(),
                "reuse", Map.of(), "scope", scope(f), "budget", Map.of("maxNodes", 6, "maxImages", 0, "maxVideoSeconds", 0, "maxParallel", 2, "maxBatchItems", 2));
        String run = start(f, ok(post(f, "/api/canvases/" + f.canvas + "/run-plans", request)));
        var waiting = awaitState(f, run, "WAITING_CONFIRMATION");
        var parent = nodeResult(waiting, batch);
        String path = "/api/canvas-runs/" + run + "/nodes/" + parent.path("id").asText() + "/batch/review";
        assertThat(nodeResult(waiting, next).path("status").asText()).isEqualTo("WAITING_INPUT");
        assertThat(children(waiting, parent.path("id").asText())).hasSize(2);
        var retry = post(f, path, Map.of("expectedRevision", parent.path("revision").asLong(), "action", "RETRY_FAILED", "note", "验证本次预算限制", "scope", scope(f)));
        assertThat(retry.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(retry.getBody().toString()).contains("上限");
        ok(post(f, path, Map.of("expectedRevision", parent.path("revision").asLong(), "action", action, "note", "确认保留已有裁剪结果", "scope", scope(f))));
        var done = awaitState(f, run, action.equals("USE_SUCCEEDED") ? "SUCCEEDED" : "PARTIAL_FAILED");
        assertThat(nodeResult(done, batch).at("/outputs/values/results")).hasSize(1);
        assertThat(nodeResult(done, next).path("status").asText()).isEqualTo(action.equals("USE_SUCCEEDED") ? "SUCCEEDED" : "BLOCKED");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_node_run WHERE run_id = ? AND parent_node_run_id IS NOT NULL", Integer.class, run)).isEqualTo(2);
    }

    @Test void executesOrderedDagAndCommitsEventsWithoutChangingTheCanvasDocument() {
        var f = fixture();
        String a = id(), b = id(), combine = id();
        create(f, List.of(node(a, "text", Map.of("content", "甲")), node(b, "text", Map.of("content", "乙")),
                node(combine, "textCombine", Map.of("separator", " / ")), node(id(), "note", Map.of("content", "不执行"))),
                List.of(edge(a, "text", combine, "texts", 1), edge(b, "text", combine, "texts", 0)));
        var plan = plan(f, "ALL", List.of(), Map.of());
        assertThat(plan.path("issues")).isEmpty();
        assertThat(plan.path("nodes")).hasSize(3);
        String run = start(f, plan);
        var done = awaitState(f, run, "SUCCEEDED");
        assertThat(nodeResult(done, combine).at("/outputs/values/text/0/text").asText()).isEqualTo("乙 / 甲");
        assertThat(snapshot(f).at("/document/revision").asLong()).isEqualTo(plan.path("documentRevision").asLong());
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM afv_canvas_run_event WHERE run_id = ?", Integer.class, run)).isGreaterThan(3));
        assertThat(done.toString()).doesNotContain("apiHash", "snapshot_json", "ownerInstanceId");
    }

    @Test void persistsSelectionAndUsesFrozenInputsAfterEditingAndDeletingTheSource() {
        var f = fixture();
        String a = id(), b = id(), choice = id(), combine = id();
        create(f, List.of(node(a, "text", Map.of("content", "原文甲")), node(b, "text", Map.of("content", "原文乙")),
                        node(choice, "selection", Map.of("valueType", "text")), node(combine, "textCombine", Map.of())),
                List.of(edge(a, "text", choice, "candidates", 0), edge(b, "text", choice, "candidates", 1), edge(choice, "selected", combine, "texts", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var waiting = awaitState(f, run, "WAITING_SELECTION");
        var waitingNode = nodeResult(waiting, choice);
        String outputId = waitingNode.at("/inputs/values/candidates/1/id").asText();
        commands(f, List.of(Map.of("type", "canvas.node.patch", "nodeId", a, "expectedConfigVersion", 0,
                "configPatch", Map.of("content", "运行后的新正文")), Map.of("type", "canvas.node.delete", "nodeId", b,
                "expectedConfigVersion", 0, "expectedLayoutVersion", 0)));
        var payload = Map.of("expectedRevision", waitingNode.path("revision").asLong(), "outputId", outputId, "scope", scope(f));
        String path = "/api/canvas-runs/" + run + "/nodes/" + waitingNode.path("id").asText() + "/select";
        ok(post(f, path, payload));
        ok(post(f, path, payload));
        var done = awaitState(f, run, "SUCCEEDED");
        assertThat(nodeResult(done, combine).at("/outputs/values/text/0/text").asText()).isEqualTo("原文乙");
        assertThat(nodeResult(done, a).path("derivedState").asText()).isEqualTo("STALE");
        assertThat(nodeResult(done, b).path("derivedState").asText()).isEqualTo("NODE_DELETED");
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_canvas_node WHERE id = ?", Boolean.class, b)).isTrue();
        assertThat(post(f, path, Map.of("expectedRevision", waitingNode.path("revision").asLong(), "outputId", "another", "scope", scope(f)))
                .getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
    }

    @Test void requiresExplicitDependenciesAndCanReuseTheExactHistoricalOutputs() {
        var f = fixture();
        String a = id(), combine = id();
        create(f, List.of(node(a, "text", Map.of("content", "固定输出")), node(combine, "textCombine", Map.of())),
                List.of(edge(a, "text", combine, "texts", 0)));
        var single = plan(f, "SINGLE", List.of(combine), Map.of());
        assertThat(single.path("issues").toString()).contains("未运行");
        var deps = plan(f, "DEPENDENCIES", List.of(combine), Map.of());
        assertThat(deps.path("nodes")).hasSize(2);
        var original = awaitState(f, start(f, deps), "SUCCEEDED");
        String originalNodeRun = nodeResult(original, a).path("id").asText();
        commands(f, List.of(Map.of("type", "canvas.node.patch", "nodeId", a, "expectedConfigVersion", 0, "configPatch", Map.of("content", "新的文本"))));
        var reuse = plan(f, "SELECTION", List.of(combine), Map.of(a, originalNodeRun));
        assertThat(reuse.path("issues")).isEmpty();
        var done = awaitState(f, start(f, reuse), "SUCCEEDED");
        assertThat(nodeResult(done, combine).at("/outputs/values/text/0/text").asText()).isEqualTo("固定输出");
        assertThat(nodeResult(done, a).path("action").asText()).isEqualTo("REUSE");
    }

    @Test void failedInputsOnlyBlockTheirDependentBranch() {
        var f = fixture();
        var scenario = textModel(f, 503, false, "stop");
        String failed = id(), blocked = id(), independent = id();
        create(f, List.of(node(failed, "textGeneration", Map.of("modelId", f.id, "prompt", "text-" + f.id)),
                        node(blocked, "textCombine", Map.of()), node(independent, "text", Map.of("content", "仍然执行"))),
                List.of(edge(failed, "text", blocked, "texts", 0)));
        var done = awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "PARTIAL_FAILED");
        assertThat(nodeResult(done, independent).path("status").asText()).isEqualTo("SUCCEEDED");
        assertThat(nodeResult(done, failed).path("status").asText()).isEqualTo("FAILED");
        assertThat(nodeResult(done, blocked).path("status").asText()).isEqualTo("BLOCKED");
        assertThat(scenario.posts).hasValue(1);
    }

    @Test void emptyTextDraftIsRejectedByThePlanBeforeExecution() {
        var f = fixture();
        String empty = id(), next = id();
        create(f, List.of(node(empty, "text", Map.of("content", "")), node(next, "textCombine", Map.of())),
                List.of(edge(empty, "text", next, "texts", 0)));
        assertThat(plan(f, "ALL", List.of(), Map.of()).path("issues").toString()).contains("NODE_INVALID", "INVALID_UPSTREAM");
    }

    @Test void admissionIsIdempotentAndWithdrawalPreventsLateStart() {
        var f = fixture();
        create(f, List.of(node(id(), "text", Map.of("content", "开始"))), List.of());
        var p = plan(f, "ALL", List.of(), Map.of());
        var request = startRequest(f, p);
        String path = "/api/canvases/" + f.canvas + "/runs";
        String first = ok(post(f, path, request)).path("id").asText();
        assertThat(ok(post(f, path, request)).path("id").asText()).isEqualTo(first);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_run WHERE canvas_id = ?", Integer.class, f.canvas)).isEqualTo(1);
        var withdrawn = startRequest(f, plan(f, "ALL", List.of(), Map.of()));
        ok(post(f, "/api/canvases/" + f.canvas + "/run-requests/" + withdrawn.get("requestId") + "/withdraw", Map.of("scope", scope(f))));
        assertThat(post(f, path, withdrawn).getStatusCode()).isEqualTo(HttpStatus.GONE);
        var stale = startRequest(f, plan(f, "ALL", List.of(), Map.of()));
        create(f, List.of(node(id(), "note", Map.of("content", "文档发生变化"))), List.of());
        assertThat(post(f, path, stale).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        var wrong = new HashMap<>(request);
        wrong.put("scope", Map.of("userId", f.id, "teamId", 900000));
        assertThat(post(f, path, wrong).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
    }

    @Test void cancellationClosesWaitingBarriersWithoutResurrectingTheirChildren() {
        var f = fixture();
        String a = id(), selection = id(), after = id();
        create(f, List.of(node(a, "text", Map.of("content", "候选")), node(selection, "selection", Map.of("valueType", "text")), node(after, "textCombine", Map.of())),
                List.of(edge(a, "text", selection, "candidates", 0), edge(selection, "selected", after, "texts", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var wait = awaitState(f, run, "WAITING_SELECTION");
        var selectionNode = nodeResult(wait, selection);
        String cancel = "/api/canvas-runs/" + run + "/cancel";
        ok(post(f, cancel, Map.of("scope", scope(f))));
        ok(post(f, cancel, Map.of("scope", scope(f))));
        var done = awaitState(f, run, "CANCELLED");
        assertThat(nodeResult(done, after).path("status").asText()).isEqualTo("CANCELLED");
        assertThat(post(f, "/api/canvas-runs/" + run + "/nodes/" + selectionNode.path("id").asText() + "/select", Map.of(
                "expectedRevision", selectionNode.path("revision").asLong(), "outputId", selectionNode.at("/inputs/values/candidates/0/id").asText(), "scope", scope(f)))
                .getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
    }

    @Test void expiredWorkerLeaseIsTakenOverAndOldWorkerCannotCommit() throws Exception {
        var f = fixture();
        String source = id();
        create(f, List.of(node(source, "text", Map.of("content", "恢复后输出"))), List.of());
        String[] runId = new String[1];
        new TransactionTemplate(manager).executeWithoutResult(status -> {
            jdbc.queryForList("SELECT id FROM afv_generation_dispatch_lock WHERE id = 1 FOR UPDATE");
            runId[0] = start(f, plan(f, "ALL", List.of(), Map.of()));
            jdbc.update("""
                    UPDATE afv_canvas_node_run SET status = 'READY', input_json = '{"values":{}}', owner_instance_id = 'lost-worker', owner_epoch = 4,
                        lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE run_id = ?
                    """, runId[0]);
        });
        String nodeRun = jdbc.queryForObject("SELECT id FROM afv_canvas_node_run WHERE run_id = ?", String.class, runId[0]);
        await().pollInterval(Duration.ofMillis(200)).atMost(Duration.ofSeconds(15)).untilAsserted(() ->
                assertThat(ok(get(f, "/api/canvas-runs/" + runId[0])).path("status").asText()).as(jdbc.queryForList(
                        "SELECT status, owner_instance_id, owner_epoch, lease_until, error_msg, next_run_at FROM afv_canvas_node_run WHERE run_id = ?", runId[0]).toString())
                        .isEqualTo("SUCCEEDED"));
        assertThat(jdbc.queryForObject("SELECT owner_epoch FROM afv_canvas_node_run WHERE id = ?", Long.class, nodeRun)).isGreaterThan(4);
        assertThatThrownBy(() -> runtime.step(new Claim(nodeRun, runId[0], "lost-worker", 4))).hasMessageContaining("接管");
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void assetsGenerateVideoDirectlyAndOnlyExplicitFramesReachTheProvider(boolean withFrame) throws Exception {
        var f = fixture();
        jdbc.update("INSERT INTO afv_asset(id,project_id,name,type,owner_type,owner_id) VALUES(?,?,'雨夜角色','character',1,?)", f.id, f.id, f.id);
        String image = upload(f, PNG, "character.png", "image/png");
        jdbc.update("INSERT INTO afv_asset_item(id,asset_id,name,item_type,properties,image_url) VALUES(?,?,'阿青','original',JSON_OBJECT('description','穿蓝色雨衣，站在灯塔门口'),?)",
                f.id, f.id, "media://" + image);
        byte[] videoBytes = CanvasAgentMediaFixtures.video(FILES.resolve("direct-assets-" + f.id), 4);
        AtomicReference<byte[]> sent = new AtomicReference<>();
        AtomicInteger posts = new AtomicInteger();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/videos", exchange -> {
            String path = exchange.getRequestURI().getPath();
            byte[] body; String type = "application/json";
            if (exchange.getRequestMethod().equals("POST") && path.equals("/v1/videos")) {
                sent.set(exchange.getRequestBody().readAllBytes()); posts.incrementAndGet();
                body = JSON.writeValueAsBytes(Map.of("id", "direct-assets", "status", "queued"));
            } else if (path.equals("/v1/videos/direct-assets")) {
                body = JSON.writeValueAsBytes(Map.of("id", "direct-assets", "status", "completed", "seconds", "4"));
            } else if (path.equals("/v1/videos/direct-assets/content")) {
                boolean thumbnail = "variant=thumbnail".equals(exchange.getRequestURI().getQuery());
                body = thumbnail ? PNG : videoBytes; type = thumbnail ? "image/png" : "video/mp4";
            } else { exchange.sendResponseHeaders(404, -1); exchange.close(); return; }
            exchange.getResponseHeaders().set("Content-Type", type); exchange.sendResponseHeaders(200, body.length);
            try (var output = exchange.getResponseBody()) { output.write(body); }
        });
        server.start();
        try {
            jdbc.update("INSERT INTO afv_api_config(id,name,platform,video_protocol,api_url,api_key) VALUES(?,'直接创作视频','openai_compatible','openai',?,'test-key')",
                    f.id, "http://127.0.0.1:" + server.getAddress().getPort());
            jdbc.update("""
                    INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels)
                    VALUES(?,'直接创作视频模型','sora-2',3,'openai',?,'{"supportFirstFrame":true,"referenceImageInputFormats":["data_uri"]}',1,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())
                    """, f.id, f.id);
            String asset = id(), generate = id();
            List<Map<String, Object>> links = new ArrayList<>();
            links.add(edge(asset, "assetItem", generate, "context", 0));
            if (withFrame) links.add(edge(asset, "image", generate, "firstFrame", 0));
            create(f, List.of(resourceNode(asset, "assetItem", f.id), node(generate, "videoGeneration",
                    Map.of("modelId", f.id, "prompt", "让角色抬头看向灯塔", "duration", 4, "count", 1))), links);
            var preview = plan(f, "ALL", List.of(), Map.of());
            assertThat(preview.path("issues")).isEmpty();
            String run = start(f, preview);
            assertThat(jdbc.queryForObject("SELECT deadline_at FROM afv_canvas_run WHERE id=?", LocalDateTime.class, run))
                    .isEqualTo(LocalDateTime.ofInstant(AgentExecutionLimits.UNLIMITED_DEADLINE, ZoneOffset.UTC));
            var done = awaitState(f, run, "SUCCEEDED");
            assertThat(nodeResult(done, generate).at("/outputs/values/videos/0/mediaId").asText()).isNotBlank();
            String body = new String(sent.get(), StandardCharsets.UTF_8);
            assertThat(body).contains("阿青", "穿蓝色雨衣", "让角色抬头看向灯塔").doesNotContain("media://");
            if (withFrame) assertThat(body).contains("input_reference");
            else assertThat(body).doesNotContain("input_reference");
            assertThat(posts).hasValue(1);
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_storyboard WHERE project_id=?", Long.class, f.id)).isZero();
        } finally { server.stop(0); }
    }

    @Test void aCollectionOfAssetsCanGenerateImagesWithoutAnIntermediateStoryboard() {
        var f = fixture(); model(f);
        long secondItem = IDS.incrementAndGet();
        jdbc.update("INSERT INTO afv_asset(id,project_id,name,type,owner_type,owner_id) VALUES(?,?,'批量角色设定','character',1,?)", f.id, f.id, f.id);
        jdbc.update("INSERT INTO afv_asset_item(id,asset_id,name,item_type,properties) VALUES(?,?,'阿青','original',JSON_OBJECT('description','蓝色雨衣')),(?,?,'阿白','variant',JSON_OBJECT('description','白色风衣'))",
                f.id, f.id, secondItem, f.id);
        String asset = id(), batch = id();
        create(f, List.of(resourceNode(asset, "asset", f.id), node(batch, "foreach", Map.of(
                "inputType", "assetItem", "resultType", "image", "maxItems", 2, "operationKind", "imageGeneration",
                "inputPort", "context", "outputPort", "images", "operationConfig", Map.of("modelId", f.id, "width", 32, "height", 32, "count", 1)))),
                List.of(edge(asset, "items", batch, "items", 0)));
        var preview = plan(f, "ALL", List.of(), Map.of());
        assertThat(preview.path("issues")).isEmpty();
        var done = awaitState(f, start(f, preview), "SUCCEEDED");
        assertThat(nodeResult(done, batch).at("/outputs/values/results")).hasSize(2);
        assertThat(done.path("nodes")).allMatch(node -> !node.path("kind").asText().startsWith("storyboard"));
    }

    @Test void generationUsesOneAtomicSubmissionAndRegisteredMediaAndExplicitReuseNeverPostsAgain() {
        var f = fixture();
        model(f);
        String image = id();
        create(f, List.of(node(image, "imageGeneration", Map.of("modelId", f.id, "prompt", "canvas-" + f.id, "width", 32, "height", 32, "count", 1))), List.of());
        var p = plan(f, "ALL", List.of(), Map.of());
        assertThat(p.path("issues")).isEmpty();
        assertThat(p.at("/cost/images").asInt()).isEqualTo(1);
        String run = start(f, p);
        var done = awaitState(f, run, "SUCCEEDED");
        var nodeRun = nodeResult(done, image);
        assertThat(nodeRun.path("derivedState").asText()).isEqualTo("CURRENT");
        String media = nodeRun.at("/outputs/values/images/0/mediaId").asText();
        assertThat(media).isNotBlank();
        assertThat(jdbc.queryForObject("SELECT state FROM afv_media_resource WHERE id = ?", String.class, media)).isEqualTo("READY");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission WHERE canvas_node_run_id = ?", Integer.class, nodeRun.path("id").asText())).isEqualTo(1);
        assertThat(POSTS.get("canvas-" + f.id).get()).isEqualTo(1);
        var reuse = plan(f, "ALL", List.of(), Map.of(image, nodeRun.path("id").asText()));
        assertThat(done.at("/statistics/media/images").asInt()).isEqualTo(1);
        assertThat(done.at("/statistics/media/bytes").asLong()).isEqualTo(PNG.length);
        String generationJob = jdbc.queryForObject("SELECT id FROM afv_remote_generation_job WHERE submission_id = ?", String.class, nodeRun.path("externalId").asText());
        jdbc.update("UPDATE afv_remote_generation_job SET input_json = JSON_SET(input_json, '$.inputMediaVersions', JSON_OBJECT(?, 1)) WHERE id = ?", media, generationJob);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE media_id = ? AND anchor_kind = 'GENERATION_INPUT' AND anchor_id = ?", Integer.class, media, generationJob)).isEqualTo(1);
        assertThat(done.at("/statistics/elapsedMs").asLong()).isGreaterThan(0);
        assertThat(reuse.at("/cost/images").asInt()).isZero();
        var repeated = awaitState(f, start(f, reuse), "SUCCEEDED");
        assertThat(nodeResult(repeated, image).at("/outputs/values/images/0/mediaId").asText()).isEqualTo(media);
        assertThat(repeated.at("/statistics/nodes/reused").asInt()).isEqualTo(1);
        assertThat(repeated.at("/statistics/media/images").asInt()).isZero();
        assertThat(POSTS.get("canvas-" + f.id).get()).isEqualTo(1);
        awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "SUCCEEDED");
        assertThat(POSTS.get("canvas-" + f.id).get()).isEqualTo(2);
    }

    @Test void twoModelBranchesSelectAndReapplyFrozenHistoricalImageWithoutRegeneration() throws Exception {
        var f = fixture();
        model(f);
        long secondModel = IDS.incrementAndGet();
        model(new Fixture(secondModel, f.canvas, f.headers));
        jdbc.update("UPDATE afv_ai_model SET config=? WHERE id IN (?,?)", JSON.writeValueAsString(Map.of(
                "defaultWidth", 32, "defaultHeight", 32, "supportReferenceImages", true,
                "referenceImageInputFormats", List.of("data_uri"), "maxReferenceImages", 1)), f.id, secondModel);
        String promptA = "journey-branch-a-" + f.id, promptB = "journey-branch-b-" + f.id;
        var red = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < 32; x++) for (int y = 0; y < 32; y++) red.setRGB(x, y, 0xff0000);
        var redBytes = new ByteArrayOutputStream();
        ImageIO.write(red, "png", redBytes);
        IMAGE_EDIT_OUTPUTS.put(promptA, PNG);
        IMAGE_EDIT_OUTPUTS.put(promptB, redBytes.toByteArray());
        String reference = upload(f, PNG, "共同首帧.png", "image/png");
        jdbc.update("INSERT INTO afv_storyboard(id,project_id,title) VALUES(?,?,'分支验收分镜')", f.id, f.id);
        jdbc.update("INSERT INTO afv_storyboard_episode(id,storyboard_id,episode_number,title) VALUES(?,?,1,'雨夜灯塔')", f.id, f.id);
        jdbc.update("INSERT INTO afv_storyboard_scene(id,storyboard_id,episode_id) VALUES(?,?,?)", f.id, f.id, f.id);
        jdbc.update("INSERT INTO afv_storyboard_item(id,storyboard_id,storyboard_episode_id,storyboard_scene_id,content) VALUES(?,?,?,?,'保持人物造型的灯塔镜头')", f.id, f.id, f.id, f.id);
        String source = id(), a = id(), b = id(), choice = id(), target = id(), write = id();
        create(f, List.of(node(source, "image", Map.of("resourceId", reference)),
                        node(a, "imageGeneration", Map.of("modelId", f.id, "prompt", promptA, "width", 32, "height", 32, "count", 1)),
                        node(b, "imageGeneration", Map.of("modelId", secondModel, "prompt", promptB, "width", 32, "height", 32, "count", 1)),
                        node(choice, "selection", Map.of("valueType", "image")), resourceNode(target, "shot", f.id),
                        node(write, "writeback", writeback("shot", "image", "firstFrameImageUrl"))),
                List.of(edge(source, "media", a, "referenceImages", 0), edge(source, "media", b, "referenceImages", 0),
                        edge(a, "images", choice, "candidates", 0), edge(b, "images", choice, "candidates", 1),
                        edge(choice, "selected", write, "value", 0), edge(target, "shot", write, "target", 0)));
        var firstPlan = plan(f, "ALL", List.of(), Map.of());
        assertThat(firstPlan.at("/cost/images").asInt()).isEqualTo(2);
        String firstRun = start(f, firstPlan);
        var firstSelection = awaitState(f, firstRun, "WAITING_SELECTION");
        var aResult = nodeResult(firstSelection, a);
        var bResult = nodeResult(firstSelection, b);
        String mediaA = aResult.at("/outputs/values/images/0/mediaId").asText();
        String mediaB = bResult.at("/outputs/values/images/0/mediaId").asText();
        assertThat(mediaA).isNotBlank().isNotEqualTo(mediaB);
        for (var branch : List.of(aResult, bResult)) {
            assertThat(branch.at("/inputs/values/referenceImages/0/mediaId").asText()).isEqualTo(reference);
        }
        for (String prompt : List.of(promptA, promptB)) {
            assertThat(POSTS.get(prompt)).hasValue(1);
            assertThat(new String(IMAGE_EDIT_REQUESTS.get(prompt), StandardCharsets.ISO_8859_1))
                    .contains(new String(PNG, StandardCharsets.ISO_8859_1));
        }
        chooseImageAndApproveShot(f, firstRun, firstSelection, choice, write, mediaB);
        assertThat(jdbc.queryForObject("SELECT first_frame_image_url FROM afv_storyboard_item WHERE id=?", String.class, f.id))
                .isEqualTo("media://" + mediaB);

        commands(f, List.of(Map.of("type", "canvas.node.patch", "nodeId", a, "expectedConfigVersion", 0,
                "configPatch", Map.of("prompt", "显式修改后的未来分支，尚未运行"))));
        var historical = ok(get(f, "/api/canvas-runs/" + firstRun));
        assertThat(nodeResult(historical, a).path("derivedState").asText()).isEqualTo("STALE");
        assertThat(nodeResult(historical, a).at("/outputs/values/images/0/mediaId").asText()).isEqualTo(mediaA);
        var reusedPlan = plan(f, "SELECTION", List.of(choice, target, write), Map.of(
                a, aResult.path("id").asText(), b, bResult.path("id").asText()));
        assertThat(reusedPlan.at("/cost/images").asInt()).isZero();
        String secondRun = start(f, reusedPlan);
        var secondSelection = awaitState(f, secondRun, "WAITING_SELECTION");
        assertThat(nodeResult(secondSelection, a).path("action").asText()).isEqualTo("REUSE");
        assertThat(nodeResult(secondSelection, b).path("action").asText()).isEqualTo("REUSE");
        chooseImageAndApproveShot(f, secondRun, secondSelection, choice, write, mediaA);
        assertThat(jdbc.queryForObject("SELECT first_frame_image_url FROM afv_storyboard_item WHERE id=?", String.class, f.id))
                .isEqualTo("media://" + mediaA);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_storyboard_item WHERE id=?", Integer.class, f.id)).isEqualTo(2);
        assertThat(POSTS.get(promptA)).hasValue(1);
        assertThat(POSTS.get(promptB)).hasValue(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_resource WHERE id IN (?,?) AND state='READY'", Integer.class, mediaA, mediaB)).isEqualTo(2);
        assertThat(nodeResult(ok(get(f, "/api/canvas-runs/" + firstRun)), b).at("/outputs/values/images/0/mediaId").asText()).isEqualTo(mediaB);
        Files.writeString(Path.of("target/g-branch-journey-results.json"), JSON.writerWithDefaultPrettyPrinter().writeValueAsString(Map.of(
                "provider", "local HTTP fixture", "samePrivateFrameDeliveredToBothModels", true,
                "firstPlanImages", 2, "historyPlanImages", 0, "modelPosts", Map.of("A", POSTS.get(promptA).get(), "B", POSTS.get(promptB).get()),
                "chosenSequence", List.of("B", "historical A"), "shotVersion", 2, "bothHistoryOutputsReady", true)));
    }

    private void chooseImageAndApproveShot(Fixture f, String run, JsonNode view, String choice, String write, String media) {
        var selection = nodeResult(view, choice);
        JsonNode candidate = null;
        for (var value : selection.at("/inputs/values/candidates")) if (value.path("mediaId").asText().equals(media)) candidate = value;
        assertThat(candidate).isNotNull();
        ok(post(f, "/api/canvas-runs/" + run + "/nodes/" + selection.path("id").asText() + "/select",
                Map.of("expectedRevision", selection.path("revision").asLong(), "outputId", candidate.path("id").asText(), "scope", scope(f))));
        var pending = nodeResult(awaitState(f, run, "WAITING_CONFIRMATION"), write);
        ok(post(f, "/api/canvas-runs/" + run + "/nodes/" + pending.path("id").asText() + "/writeback", decision(f, pending, "APPROVE")));
        awaitState(f, run, "SUCCEEDED");
    }

    @Test void imageProcessingPreservesOrderedDependenciesAndRegistersRealImageDimensions() {
        var f = fixture();
        String media = upload(f, PNG, "source.png", "image/png");
        String source = id(), resize = id(), crop = id();
        create(f, List.of(node(source, "image", Map.of("resourceId", media)),
                        node(resize, "imageTransform", Map.of("operation", "resize", "width", 24, "height", 16, "fit", "contain")),
                        node(crop, "imageTransform", Map.of("operation", "crop", "x", 4, "y", 2, "width", 8, "height", 8))),
                List.of(edge(source, "media", resize, "image", 0), edge(resize, "image", crop, "image", 0)));
        var done = awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "SUCCEEDED");
        assertThat(nodeResult(done, resize).at("/outputs/values/image/0/data/content/width").asInt()).isEqualTo(24);
        assertThat(nodeResult(done, resize).at("/outputs/values/image/0/data/content/height").asInt()).isEqualTo(16);
        assertThat(nodeResult(done, crop).at("/outputs/values/image/0/data/content/width").asInt()).isEqualTo(8);
        assertThat(nodeResult(done, source).path("derivedState").asText()).isEqualTo("CURRENT");
        assertThat(done.at("/statistics/media/images").asInt()).isEqualTo(2);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_media_job m JOIN afv_media_job j ON j.id = m.job_id WHERE j.actor_user_id = ? AND j.status = 'SUCCEEDED'", Integer.class, f.id)).isEqualTo(2);
    }

    @Test void composeFrameExtractionAndMixingUseRealFfmpegAndManagedOutputs() throws Exception {
        var f = fixture();
        String videoA = upload(f, video(f, "red", true), "red.mp4", "video/mp4");
        String videoB = upload(f, video(f, "blue", false), "blue.mp4", "video/mp4");
        String music = upload(f, wav(), "music.wav", "audio/wav");
        String a = id(), b = id(), audio = id(), compose = id(), frames = id(), mix = id();
        create(f, List.of(node(a, "video", Map.of("resourceId", videoA)), node(b, "video", Map.of("resourceId", videoB)),
                        node(audio, "audio", Map.of("resourceId", music)), node(compose, "videoCompose", Map.of("resolution", "128x96", "frameRate", 24)),
                        node(frames, "frameExtract", Map.of("frameType", "both")), node(mix, "audioMix", Map.of("musicVolume", 0.2, "originalVolume", 0.5))),
                List.of(edge(a, "media", compose, "videos", 0), edge(b, "media", compose, "videos", 1),
                        edge(compose, "video", frames, "video", 0), edge(compose, "video", mix, "video", 0), edge(audio, "media", mix, "music", 0)));
        var done = awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "SUCCEEDED");
        var result = nodeResult(done, compose).at("/outputs/values/video/0/data/content");
        assertThat(result.path("width").asInt()).isEqualTo(128);
        assertThat(result.path("durationMs").asLong()).isBetween(5900L, 6300L);
        assertThat(nodeResult(done, frames).at("/outputs/values/firstFrame")).hasSize(1);
        assertThat(nodeResult(done, frames).at("/outputs/values/lastFrame")).hasSize(1);
        assertThat(nodeResult(done, mix).at("/outputs/values/video/0/data/content/hasAudio").asBoolean()).isTrue();
        assertThat(done.at("/statistics/media/images").asInt()).isEqualTo(2);
        assertThat(done.at("/statistics/media/videos").asInt()).isEqualTo(2);
        assertThat(done.at("/statistics/media/videoDurationMs").asLong()).isBetween(11800L, 12600L);
        assertThat(done.at("/statistics/media/audios").asInt()).isZero();
    }

    @Test void audioCancellationKeepsCapacityUntilTheOriginalRemoteTaskActuallyFinishes() throws Exception {
        var f = fixture();
        var scenario = new AudioScenario(false);
        AUDIO.put("audio-" + f.id, scenario);
        audioModel(f);
        String source = id(), music = id();
        String media = upload(f, video(f, "green", false), "source.mp4", "video/mp4");
        create(f, List.of(node(source, "video", Map.of("resourceId", media)), node(music, "musicGeneration", Map.of("apiConfigId", f.id, "operation", "music", "prompt", "audio-" + f.id))),
                List.of(edge(source, "media", music, "video", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        await().atMost(Duration.ofSeconds(20)).until(() -> scenario.posts.get() == 1);
        String job = jdbc.queryForObject("SELECT external_id FROM afv_canvas_node_run WHERE run_id = ? AND node_id = ?", String.class, run, music);
        ok(post(f, "/api/canvas-runs/" + run + "/cancel", Map.of("scope", scope(f))));
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject("SELECT cancel_requested FROM afv_media_job WHERE id = ?", Boolean.class, job)).isTrue());
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_media_job WHERE job_id = ?", Boolean.class, job)).isTrue();
        assertThat(ok(get(f, "/api/canvas-runs/" + run)).path("status").asText()).isNotEqualTo("CANCELLED");
        scenario.complete.set(true);
        awaitState(f, run, "CANCELLED");
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_media_job WHERE job_id = ?", Boolean.class, job)).isFalse();
        assertThat(scenario.posts).hasValue(1);
    }

    @Test void canvasAndEpisodeAudioShareOneApiSlotAndCancellationWaitsForItsRealHttpResponse() throws Exception {
        var f = fixture();
        audioModel(f);
        jdbc.update("UPDATE afv_api_config SET status = 0 WHERE platform = 'sonilo' AND id <> ?", f.id);
        caches.getCache("episodeBgmConfig").clear();
        String episodeKey = "episode-audio-" + f.id, canvasKey = "canvas-audio-" + f.id;
        var episodeAudio = new AudioScenario(false);
        var canvasAudio = new AudioScenario(false);
        AUDIO.put(episodeKey, episodeAudio);
        AUDIO.put(canvasKey, canvasAudio);
        String previousPrompt = bgmProperties.getMusicPrompt();
        bgmProperties.setMusicPrompt(episodeKey);
        try {
            String media = upload(f, video(f, "violet", false), "audio-source.mp4", "video/mp4");
            long board = IDS.incrementAndGet(), episode = IDS.incrementAndGet(), shot = IDS.incrementAndGet();
            jdbc.update("INSERT INTO afv_storyboard(id, project_id) VALUES (?, ?)", board, f.id);
            jdbc.update("INSERT INTO afv_storyboard_episode(id, storyboard_id, title) VALUES (?, ?, '音频共享额度验证')", episode, board);
            jdbc.update("INSERT INTO afv_storyboard_item(id, storyboard_id, storyboard_episode_id, video_url) VALUES (?, ?, ?, ?)",
                    shot, board, episode, "media://" + media);
            String job = ok(post(f, "/api/storyboard/episode/" + episode + "/compose-video", Map.of(
                    "version", 0, "requestId", id(), "includeBgm", true))).asText();
            await().atMost(Duration.ofSeconds(25)).until(() -> episodeAudio.posts.get() == 1);
            String source = id(), music = id();
            create(f, List.of(node(source, "video", Map.of("resourceId", media)), node(music, "musicGeneration",
                    Map.of("apiConfigId", f.id, "operation", "music", "prompt", canvasKey))), List.of(edge(source, "media", music, "video", 0)));
            String run = start(f, plan(f, "ALL", List.of(), Map.of()));
            await().atMost(Duration.ofSeconds(15)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                    "SELECT COUNT(*) FROM afv_canvas_media_job WHERE api_config_id = ?", Integer.class, f.id)).isEqualTo(1));
            assertThat(canvasAudio.posts).hasValue(0);
            ok(post(f, "/api/media-jobs/" + job + "/cancel", Map.of()));
            await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(ok(get(f, "/api/media-jobs/" + job)).path("phase").asText()).contains("等待原音频请求结束"));
            assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_episode_audio_job WHERE job_id = ?", Boolean.class, job)).isTrue();
            assertThat(canvasAudio.posts).hasValue(0);
            episodeAudio.complete.set(true);
            await().atMost(Duration.ofSeconds(20)).untilAsserted(() -> assertThat(ok(get(f, "/api/media-jobs/" + job)).path("status").asText()).isEqualTo("CANCELLED"));
            await().atMost(Duration.ofSeconds(15)).until(() -> canvasAudio.posts.get() == 1);
            assertThat(jdbc.queryForObject("SELECT bgm_audio_url FROM afv_storyboard_episode WHERE id = ?", String.class, episode)).isNull();
            int version = jdbc.queryForObject("SELECT version FROM afv_storyboard_episode WHERE id = ?", Integer.class, episode);
            String second = ok(post(f, "/api/storyboard/episode/" + episode + "/compose-video", Map.of(
                    "version", version, "requestId", id(), "includeBgm", true))).asText();
            // 画布仍占用同一 API 时，分集领取器也不能为第二次合成开始付费调用。
            assertThat(ok(get(f, "/api/media-jobs/" + second)).path("status").asText()).isEqualTo("QUEUED");
            assertThat(mediaJobs.claim("blocked-episode-probe")).isEmpty();
            ok(post(f, "/api/media-jobs/" + second + "/cancel", Map.of()));
            await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(ok(get(f, "/api/media-jobs/" + second)).path("status").asText()).isEqualTo("CANCELLED"));
            canvasAudio.complete.set(true);
            awaitState(f, run, "SUCCEEDED");
            assertThat(episodeAudio.posts).hasValue(1);
            assertThat(canvasAudio.posts).hasValue(1);
        } finally {
            episodeAudio.complete.set(true);
            canvasAudio.complete.set(true);
            bgmProperties.setMusicPrompt(previousPrompt);
            caches.getCache("episodeBgmConfig").clear();
        }
    }

    @Test void explicitAudioApiRejectionEndsTheBranchAndReleasesItsPermit() throws Exception {
        var f = fixture();
        var scenario = new AudioScenario(false);
        scenario.rejected = true;
        AUDIO.put("audio-" + f.id, scenario);
        audioModel(f);
        String source = id(), music = id();
        String media = upload(f, video(f, "white", false), "source.mp4", "video/mp4");
        create(f, List.of(node(source, "video", Map.of("resourceId", media)), node(music, "musicGeneration",
                Map.of("apiConfigId", f.id, "operation", "music", "prompt", "audio-" + f.id))),
                List.of(edge(source, "media", music, "video", 0)));
        var failed = awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "PARTIAL_FAILED");
        var result = nodeResult(failed, music);
        assertThat(result.path("status").asText()).isEqualTo("FAILED");
        assertThat(result.path("error").asText()).contains("HTTP 503", "audio queue is full");
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_media_job WHERE job_id = ?", Boolean.class,
                result.path("externalId").asText())).isFalse();
        assertThat(scenario.posts).hasValue(1);
    }

    @Test void anUnknownAudioReceiptCanBeRecordedAndRecoveredWithoutSubmittingAgain() throws Exception {
        var f = fixture();
        var scenario = new AudioScenario(true);
        AUDIO.put("audio-" + f.id, scenario);
        audioModel(f);
        String source = id(), music = id();
        String media = upload(f, video(f, "white", false), "source.mp4", "video/mp4");
        create(f, List.of(node(source, "video", Map.of("resourceId", media)), node(music, "sfxGeneration", Map.of("apiConfigId", f.id, "operation", "sfx", "prompt", "audio-" + f.id))),
                List.of(edge(source, "media", music, "video", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var unknown = awaitState(f, run, "UNKNOWN_SUBMISSION");
        var nodeRun = nodeResult(unknown, music);
        String path = "/api/canvas-runs/" + run + "/nodes/" + nodeRun.path("id").asText() + "/media";
        var detail = ok(get(f, path));
        assertThat(detail.path("actions").toString()).contains("RECORD_RECEIPT");
        scenario.complete.set(true);
        ok(post(f, path + "/review", Map.of("expectedRevision", detail.path("revision").asLong(), "action", "RECORD_RECEIPT",
                "providerTaskId", "audio-" + f.id, "note", "测试供应商已确认原任务标识", "scope", scope(f))));
        var done = awaitState(f, run, "SUCCEEDED");
        assertThat(nodeResult(done, music).at("/outputs/values/audio/0/data/content/kind").asText()).isEqualTo("AUDIO");
        assertThat(scenario.posts).hasValue(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_job_event WHERE job_id = ? AND event_type = 'CANVAS_MEDIA_REVIEW'", Integer.class,
                nodeRun.path("externalId").asText())).isEqualTo(1);
    }

    @Test void textGenerationPublishesOnePersistentResultAndContinuesTheDag() {
        var f = fixture();
        var scenario = textModel(f, 200, false, "stop");
        String source = id(), generate = id(), combine = id();
        create(f, List.of(node(source, "text", Map.of("content", "text-" + f.id)),
                node(generate, "textGeneration", Map.of("modelId", f.id, "instruction", "只输出场景描述")),
                node(combine, "textCombine", Map.of())), List.of(edge(source, "text", generate, "prompt", 0), edge(generate, "text", combine, "texts", 0)));
        var done = awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "SUCCEEDED");
        assertThat(nodeResult(done, combine).at("/outputs/values/text/0/text").asText()).isEqualTo("模型返回文本");
        assertThat(nodeResult(done, generate).path("derivedState").asText()).isEqualTo("CURRENT");
        assertThat(scenario.posts).hasValue(1);
        assertThat(scenario.request.path("messages").get(0).path("role").asText()).isEqualTo("system");
        String job = nodeResult(done, generate).path("externalId").asText();
        assertThat(done.at("/statistics/tokens/submittedCalls").asInt()).isEqualTo(1);
        assertThat(done.at("/statistics/tokens/measuredCalls").asInt()).isEqualTo(1);
        assertThat(done.at("/statistics/tokens/input").asLong()).isEqualTo(10);
        assertThat(done.at("/statistics/tokens/output").asLong()).isEqualTo(5);
        assertThat(done.at("/statistics/tokens/total").asLong()).isEqualTo(15);
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_text_job WHERE id = ?", Boolean.class, job)).isFalse();
        assertThat(jdbc.queryForObject("SELECT request_started_at IS NOT NULL AND received_at IS NOT NULL FROM afv_canvas_text_job WHERE id = ?", Boolean.class, job)).isTrue();
    }

    @Test void explicitTextApiRejectionFailsWithoutReviewAndReleasesModelCapacity() {
        var f = fixture();
        var scenario = textModel(f, 503, false, "stop");
        String generate = id();
        create(f, List.of(node(generate, "textGeneration", Map.of("modelId", f.id, "prompt", "text-" + f.id))), List.of());
        var failed = awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "FAILED");
        var node = nodeResult(failed, generate);
        assertThat(node.path("error").asText()).contains("HTTP 503", "text queue is full");
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_text_job WHERE id = ?", Boolean.class,
                node.path("externalId").asText())).isFalse();
        assertThat(scenario.posts).hasValue(1);
        scenario.status = 200;
        awaitState(f, start(f, plan(f, "ALL", List.of(), Map.of())), "SUCCEEDED");
        assertThat(scenario.posts).hasValue(2);
    }

    @Test void uncertainTextCallHoldsCapacityUntilExplicitReviewAndNeverReposts() {
        var f = fixture();
        var scenario = textModel(f, 0, false, "stop");
        String generate = id();
        create(f, List.of(node(generate, "textGeneration", Map.of("modelId", f.id, "prompt", "text-" + f.id))), List.of());
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var unknown = awaitState(f, run, "UNKNOWN_SUBMISSION");
        var nodeRun = nodeResult(unknown, generate);
        String path = "/api/canvas-runs/" + run + "/nodes/" + nodeRun.path("id").asText() + "/text";
        var detail = ok(get(f, path));
        assertThat(detail.path("remoteCapacityHeld").asBoolean()).isTrue();
        assertThat(detail.path("actions").toString()).contains("CONFIRM_FINISHED").doesNotContain("USE_RECEIVED");
        ok(post(f, path + "/review", Map.of("expectedRevision", detail.path("revision").asLong(), "action", "CONFIRM_FINISHED",
                "note", "已在本地测试供应商确认原调用结束", "scope", scope(f))));
        awaitState(f, run, "FAILED");
        assertThat(scenario.posts).hasValue(1);
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_text_job WHERE id = ?", Boolean.class, nodeRun.path("externalId").asText())).isFalse();
        await().atMost(Duration.ofSeconds(5)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM afv_canvas_text_event WHERE job_id = ? AND event_type = 'CANVAS_TEXT_REVIEW'", Integer.class,
                nodeRun.path("externalId").asText())).isEqualTo(1));
    }

    @Test void aLateTextResponseSurvivesLeaseHandoffButRequiresReviewBeforePublishing() throws Exception {
        var f = fixture();
        var scenario = textModel(f, 200, true, "stop");
        String generate = id();
        create(f, List.of(node(generate, "textGeneration", Map.of("modelId", f.id, "prompt", "text-" + f.id))), List.of());
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        try {
            assertThat(scenario.entered.await(20, TimeUnit.SECONDS)).isTrue();
            String job = jdbc.queryForObject("SELECT id FROM afv_canvas_text_job WHERE run_id = ?", String.class, run);
            jdbc.update("UPDATE afv_canvas_text_job SET lease_until = TIMESTAMPADD(SECOND, -1, UTC_TIMESTAMP(6)) WHERE id = ?", job);
            var unknown = awaitState(f, run, "UNKNOWN_SUBMISSION");
            scenario.release.countDown();
            await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                    "SELECT result_json IS NOT NULL FROM afv_canvas_text_job WHERE id = ?", Boolean.class, job)).isTrue());
            assertThat(jdbc.queryForObject("SELECT status FROM afv_canvas_text_job WHERE id = ?", String.class, job)).isEqualTo("UNKNOWN_SUBMISSION");
            String path = "/api/canvas-runs/" + run + "/nodes/" + nodeResult(unknown, generate).path("id").asText() + "/text";
            var detail = ok(get(f, path));
            assertThat(detail.path("receivedText").asText()).isEqualTo("模型返回文本");
            ok(post(f, path + "/review", Map.of("expectedRevision", detail.path("revision").asLong(), "action", "USE_RECEIVED",
                    "note", "确认使用原调用的迟到响应", "scope", scope(f))));
            var done = awaitState(f, run, "SUCCEEDED");
            assertThat(nodeResult(done, generate).at("/outputs/values/text/0/text").asText()).isEqualTo("模型返回文本");
            assertThat(scenario.posts).hasValue(1);
        } finally { scenario.release.countDown(); }
    }

    @Test void truncatedTextIsPreservedAndCanOnlyContinueAfterTheUserAdoptsIt() {
        var f = fixture();
        var scenario = textModel(f, 200, false, "length");
        String generate = id(), next = id();
        create(f, List.of(node(generate, "textGeneration", Map.of("modelId", f.id, "prompt", "text-" + f.id)),
                node(next, "textCombine", Map.of())), List.of(edge(generate, "text", next, "texts", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var waiting = awaitState(f, run, "NEEDS_REVIEW");
        assertThat(nodeResult(waiting, next).path("status").asText()).isEqualTo("WAITING_INPUT");
        String path = "/api/canvas-runs/" + run + "/nodes/" + nodeResult(waiting, generate).path("id").asText() + "/text";
        var detail = ok(get(f, path));
        assertThat(detail.path("completeResponse").asBoolean()).isFalse();
        assertThat(detail.path("remoteCapacityHeld").asBoolean()).isFalse();
        ok(post(f, path + "/review", Map.of("expectedRevision", detail.path("revision").asLong(), "action", "USE_RECEIVED",
                "note", "已检查截断内容，本次明确采用", "scope", scope(f))));
        var done = awaitState(f, run, "SUCCEEDED");
        assertThat(nodeResult(done, next).at("/outputs/values/text/0/text").asText()).isEqualTo("模型返回文本");
        assertThat(scenario.posts).hasValue(1);
    }

    @Test void cancellingAnActiveTextCallWaitsForItsResponseAndDoesNotPublishIt() throws Exception {
        var f = fixture();
        var scenario = textModel(f, 200, true, "stop");
        jdbc.update("UPDATE afv_ai_model SET support_vision = TRUE, multimodal_input_types = JSON_ARRAY('image'), multimodal_input_transports = JSON_OBJECT('image', JSON_ARRAY('url')) WHERE id = ?", f.id);
        system.setValue(SystemConfigService.RESOURCE_BASE_URL_KEY, http.getRootUri());
        String media = upload(f, PNG, "text-input.png", "image/png");
        String generate = id(), image = id();
        create(f, List.of(node(image, "image", Map.of("resourceId", media)),
                node(generate, "textGeneration", Map.of("modelId", f.id, "prompt", "text-" + f.id))),
                List.of(edge(image, "media", generate, "images", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        try {
            assertThat(scenario.entered.await(20, TimeUnit.SECONDS)).isTrue();
            String job = jdbc.queryForObject("SELECT id FROM afv_canvas_text_job WHERE run_id = ?", String.class, run);
            String inputUrl = scenario.request.findValue("image_url").path("url").asText();
            assertThat(inputUrl).contains("/api/media-content/" + media);
            assertThat(http.getForEntity(inputUrl, byte[].class).getBody()).isEqualTo(PNG);
            ok(post(f, "/api/canvas-runs/" + run + "/cancel", Map.of("scope", scope(f))));
            await().atMost(Duration.ofSeconds(5)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                    "SELECT cancel_requested FROM afv_canvas_text_job WHERE id = ?", Boolean.class, job)).isTrue());
            assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_text_job WHERE id = ?", Boolean.class, job)).isTrue();
            assertThat(http.getForEntity(inputUrl, byte[].class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
            scenario.release.countDown();
            var done = awaitState(f, run, "CANCELLED");
            assertThat(nodeResult(done, generate).at("/outputs/values")).isEmpty();
            assertThat(done.at("/statistics/tokens/submittedCalls").asInt()).isEqualTo(1);
            assertThat(done.at("/statistics/tokens/total").asLong()).isEqualTo(15);
            assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_text_job WHERE id = ?", Boolean.class, job)).isFalse();
            assertThat(scenario.posts).hasValue(1);
        } finally { scenario.release.countDown(); }
    }

    @Test void revokedTextOwnerCanBeReconciledOnlyByFreshAdminWithVersionedIdempotentEvidence() {
        var f = fixture();
        var admin = recoveryAdmin();
        var scenario = textModel(f, 0, false, "stop");
        String text = id();
        create(f, List.of(node(text, "textGeneration", Map.of("modelId", f.id, "prompt", "text-" + f.id))), List.of());
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var unknown = awaitState(f, run, "UNKNOWN_SUBMISSION");
        String job = nodeResult(unknown, text).path("externalId").asText();
        String path = "/api/system/execution-recovery/TEXT/" + job;
        assertThat(get(f, "/api/system/execution-recovery").getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var before = ok(get(admin, path));
        assertThat(before.path("canResolve").asBoolean()).isFalse();
        var activeDecision = recoveryDecision(admin, before, id());
        assertThat(post(admin, path + "/confirm-finished", activeDecision).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        jdbc.update("UPDATE sys_user SET status = 0 WHERE id = ?", f.id);
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT status FROM afv_canvas_run WHERE id = ?", String.class, run)).isEqualTo("NEEDS_REVIEW"));
        var pending = ok(get(admin, path));
        assertThat(pending.path("canResolve").asBoolean()).isTrue();
        assertThat(pending.toString()).doesNotContain("text-" + f.id, "test-key", "input_json", "result_json", "providerHash");
        var decision = recoveryDecision(admin, pending, id());
        var missingEvidence = new HashMap<>(decision);
        missingEvidence.put("providerFinished", false);
        assertThat(post(admin, path + "/confirm-finished", missingEvidence).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        var stale = new HashMap<>(decision);
        stale.put("expectedRevision", pending.path("revision").asLong() - 1);
        assertThat(post(admin, path + "/confirm-finished", stale).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        var receipt = ok(post(admin, path + "/confirm-finished", decision));
        assertThat(ok(post(admin, path + "/confirm-finished", decision))).isEqualTo(receipt);
        var changed = new HashMap<>(decision); changed.put("note", "改写同一决定");
        assertThat(post(admin, path + "/confirm-finished", changed).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT status FROM afv_canvas_run WHERE id = ?", String.class, run)).isEqualTo("CANCELLED"));
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_text_job WHERE id = ?", Boolean.class, job)).isFalse();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_execution_recovery_decision WHERE job_id = ?", Integer.class, job)).isEqualTo(1);
        assertThat(ok(get(admin, "/api/system/execution-recovery/history")).toString()).contains(receipt.path("decisionId").asText());
        assertThat(scenario.posts).hasValue(1);
        jdbc.update("UPDATE sys_user_role SET deleted = TRUE WHERE user_id = ?", admin.id);
        assertThat(get(admin, "/api/system/execution-recovery").getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void revokedBatchWaitsForEveryOriginalChildAndNeverPublishesAdminAdoptedOutput() {
        var f = fixture();
        var admin = recoveryAdmin();
        var scenario = textModel(f, 0, false, "stop");
        jdbc.update("UPDATE afv_ai_model SET max_concurrency = 2 WHERE id = ?", f.id);
        String first = id(), second = id(), batch = id();
        create(f, List.of(node(first, "text", Map.of("content", "text-" + f.id + " A")), node(second, "text", Map.of("content", "text-" + f.id + " B")),
                        node(batch, "foreach", batch("textGeneration", Map.of("modelId", f.id), "prompt", "text", "text", "text", 2))),
                List.of(edge(first, "text", batch, "items", 0), edge(second, "text", batch, "items", 1)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        await().atMost(Duration.ofSeconds(20)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM afv_canvas_text_job WHERE run_id = ? AND status = 'UNKNOWN_SUBMISSION'", Integer.class, run)).isEqualTo(2));
        jdbc.update("UPDATE sys_user SET status = 0 WHERE id = ?", f.id);
        var jobs = jdbc.queryForList("SELECT id FROM afv_canvas_text_job WHERE run_id = ? ORDER BY id", String.class, run);
        recover(admin, "TEXT", jobs.getFirst());
        assertThat(jdbc.queryForObject("SELECT status FROM afv_canvas_run WHERE id = ?", String.class, run)).isNotEqualTo("CANCELLED");
        recover(admin, "TEXT", jobs.getLast());
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT status FROM afv_canvas_run WHERE id = ?", String.class, run)).isEqualTo("CANCELLED"));
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_node_run WHERE run_id = ? AND node_id = ? AND status <> 'CANCELLED'", Integer.class, run, batch)).isZero();
        assertThat(jdbc.queryForObject("SELECT JSON_LENGTH(output_json, '$.values') FROM afv_canvas_node_run WHERE run_id = ? AND node_id = ? AND parent_node_run_id IS NULL", Integer.class, run, batch)).isZero();
        assertThat(scenario.posts).hasValue(2);
    }

    @Test void revokedImageGenerationReleasesOnlyConfirmedJobAndPreservesOriginalExecutionIdentity() {
        var f = fixture();
        var admin = recoveryAdmin();
        model(f);
        String generate = id(), prompt = "recovery-unknown-" + f.id;
        create(f, List.of(node(generate, "imageGeneration", Map.of("modelId", f.id, "prompt", prompt))), List.of());
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var unknown = awaitState(f, run, "UNKNOWN_SUBMISSION");
        String submission = nodeResult(unknown, generate).path("externalId").asText();
        String job = jdbc.queryForObject("SELECT id FROM afv_remote_generation_job WHERE submission_id = ?", String.class, submission);
        String context = jdbc.queryForObject("SELECT context_json FROM afv_generation_submission WHERE id = ?", String.class, submission);
        jdbc.update("UPDATE sys_user SET status = 0 WHERE id = ?", f.id);
        String path = "/api/system/execution-recovery/GENERATION/" + job;
        var beforeReceipt = ok(get(admin, path));
        String attempt = jdbc.queryForObject("SELECT active_attempt_id FROM afv_remote_generation_job WHERE id = ?", String.class, job);
        generationAttempts.acknowledge(attempt, GenerationProviderReceipt.accepted("late-accepted-original"));
        // 迟到回执本身不递增旧执行行版本，所以还必须绑定完整证据摘要。
        assertThat(post(admin, path + "/confirm-finished", recoveryDecision(admin, beforeReceipt, id())).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        recover(admin, "GENERATION", job);
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject("SELECT status FROM afv_canvas_run WHERE id = ?", String.class, run)).isEqualTo("CANCELLED"));
        assertThat(jdbc.queryForObject("SELECT context_json FROM afv_generation_submission WHERE id = ?", String.class, submission)).isEqualTo(context);
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_remote_generation_job WHERE id = ?", Boolean.class, job)).isFalse();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_image_item i JOIN afv_generation_submission s ON s.task_id = i.task_id WHERE s.id = ? AND i.image_url IS NOT NULL", Integer.class, submission)).isZero();
        assertThat(POSTS.get(prompt)).hasValue(1);
    }

    @Test void revokedCanvasAudioCanEndWithoutReadingOrAdoptingItsPrivateInputs() throws Exception {
        var f = fixture();
        var admin = recoveryAdmin();
        var scenario = new AudioScenario(true);
        AUDIO.put("recovery-audio-" + f.id, scenario);
        audioModel(f);
        String source = id(), music = id();
        String media = upload(f, video(f, "gray", false), "source.mp4", "video/mp4");
        create(f, List.of(node(source, "video", Map.of("resourceId", media)), node(music, "musicGeneration", Map.of("apiConfigId", f.id, "operation", "music", "prompt", "recovery-audio-" + f.id))),
                List.of(edge(source, "media", music, "video", 0)));
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        var unknown = awaitState(f, run, "UNKNOWN_SUBMISSION");
        String job = nodeResult(unknown, music).path("externalId").asText();
        jdbc.update("UPDATE sys_user SET status = 0 WHERE id = ?", f.id);
        var view = ok(get(admin, "/api/system/execution-recovery/CANVAS_MEDIA/" + job));
        assertThat(view.toString()).doesNotContain(media, "source.mp4", "recovery-audio-" + f.id);
        recover(admin, "CANVAS_MEDIA", job);
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject("SELECT status FROM afv_canvas_run WHERE id = ?", String.class, run)).isEqualTo("CANCELLED"));
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_media_job WHERE job_id = ?", Boolean.class, job)).isFalse();
        assertThat(scenario.posts).hasValue(1);
    }

    @Test void revokedEpisodeAudioIsClosedWithoutImpersonatingBusinessWriterAndNewAuthorizedComposeCanProceed() throws Exception {
        var f = fixture();
        var admin = recoveryAdmin();
        audioModel(f);
        jdbc.update("UPDATE afv_api_config SET status = 0 WHERE platform = 'sonilo' AND id <> ?", f.id);
        caches.getCache("episodeBgmConfig").clear();
        String key = "orphan-episode-" + f.id;
        var scenario = new AudioScenario(true);
        AUDIO.put(key, scenario);
        String previousPrompt = bgmProperties.getMusicPrompt();
        bgmProperties.setMusicPrompt(key);
        try {
            String media = upload(f, video(f, "yellow", false), "episode-source.mp4", "video/mp4");
            long board = IDS.incrementAndGet(), episode = IDS.incrementAndGet(), shot = IDS.incrementAndGet();
            jdbc.update("INSERT INTO afv_storyboard(id, project_id) VALUES (?, ?)", board, f.id);
            jdbc.update("INSERT INTO afv_storyboard_episode(id, storyboard_id, title) VALUES (?, ?, '撤权分集合成验证')", episode, board);
            jdbc.update("INSERT INTO afv_storyboard_item(id, storyboard_id, storyboard_episode_id, video_url) VALUES (?, ?, ?, ?)", shot, board, episode, "media://" + media);
            String job = ok(post(f, "/api/storyboard/episode/" + episode + "/compose-video", Map.of("version", 0, "requestId", id(), "includeBgm", true))).asText();
            await().atMost(Duration.ofSeconds(20)).untilAsserted(() -> assertThat(ok(get(f, "/api/media-jobs/" + job)).path("status").asText()).isEqualTo("NEEDS_REVIEW"));
            String context = jdbc.queryForObject("SELECT context_json FROM afv_media_job WHERE id = ?", String.class, job);
            jdbc.update("UPDATE sys_user SET status = 0 WHERE id = ?", f.id);
            scenario.complete.set(true);
            recover(admin, "EPISODE_AUDIO", job);
            assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_episode_audio_job WHERE job_id = ?", Boolean.class, job)).isFalse();
            assertThat(jdbc.queryForObject("SELECT context_json FROM afv_media_job WHERE id = ?", String.class, job)).isEqualTo(context);
            assertThat(jdbc.queryForObject("SELECT version FROM afv_storyboard_episode WHERE id = ?", Integer.class, episode)).isEqualTo(1);
            assertThat(jdbc.queryForObject("SELECT composed_video_url FROM afv_storyboard_episode WHERE id = ?", String.class, episode)).isNull();
            assertThat(jdbc.queryForObject("SELECT compose_status FROM afv_storyboard_episode WHERE id = ?", Integer.class, episode)).isEqualTo(1);
            jdbc.update("UPDATE sys_user SET status = 1 WHERE id = ?", f.id);
            String next = ok(post(f, "/api/storyboard/episode/" + episode + "/compose-video", Map.of("version", 1, "requestId", id(), "includeBgm", false))).asText();
            await().atMost(Duration.ofSeconds(20)).untilAsserted(() -> assertThat(ok(get(f, "/api/media-jobs/" + next)).path("status").asText()).isEqualTo("SUCCEEDED"));
            assertThat(scenario.posts).hasValue(1);
        } finally {
            bgmProperties.setMusicPrompt(previousPrompt);
            caches.getCache("episodeBgmConfig").clear();
        }
    }

    @Test void responseArrivingAfterRevocationIsAuditedButAdminCannotAdoptItIntoTheCanvas() throws Exception {
        var f = fixture();
        var admin = recoveryAdmin();
        var scenario = textModel(f, 200, true, "stop");
        String text = id();
        create(f, List.of(node(text, "textGeneration", Map.of("modelId", f.id, "prompt", "text-" + f.id))), List.of());
        String run = start(f, plan(f, "ALL", List.of(), Map.of()));
        try {
            assertThat(scenario.entered.await(20, TimeUnit.SECONDS)).isTrue();
            jdbc.update("UPDATE sys_user SET status = 0 WHERE id = ?", f.id);
            scenario.release.countDown();
            await().atMost(Duration.ofSeconds(12)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                    "SELECT status FROM afv_canvas_text_job WHERE run_id = ?", String.class, run)).isEqualTo("NEEDS_REVIEW"));
            String job = jdbc.queryForObject("SELECT id FROM afv_canvas_text_job WHERE run_id = ?", String.class, run);
            var view = ok(get(admin, "/api/system/execution-recovery/TEXT/" + job));
            assertThat(view.path("remote").get(0).path("responseRecorded").asBoolean()).isTrue();
            assertThat(view.toString()).doesNotContain("模型返回文本", "text-" + f.id);
            recover(admin, "TEXT", job);
            await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject("SELECT status FROM afv_canvas_run WHERE id = ?", String.class, run)).isEqualTo("CANCELLED"));
            assertThat(jdbc.queryForObject("SELECT JSON_LENGTH(output_json, '$.values') FROM afv_canvas_node_run WHERE run_id = ? AND node_id = ?", Integer.class, run, text)).isZero();
            assertThat(jdbc.queryForObject("SELECT result_json IS NOT NULL FROM afv_canvas_text_job WHERE id = ?", Boolean.class, job)).isTrue();
            assertThat(scenario.posts).hasValue(1);
        } finally { scenario.release.countDown(); }
    }

    private Fixture recoveryAdmin() {
        long user = IDS.incrementAndGet();
        jdbc.update("INSERT INTO sys_user(id, username, password) VALUES (?, ?, 'test-only')", user, "recovery-admin-" + user);
        jdbc.update("INSERT INTO sys_user_role(user_id, role_id) SELECT ?, id FROM sys_role WHERE UPPER(code) = 'ADMIN' AND deleted = FALSE", user);
        var headers = new HttpHeaders();
        headers.setBearerAuth(tokens.createToken(user, "recovery-admin-" + user, null).getAccessToken());
        headers.setContentType(MediaType.APPLICATION_JSON);
        return new Fixture(user, null, headers);
    }
    private Map<String, Object> recoveryDecision(Fixture admin, JsonNode target, String request) {
        return Map.of("adminUserId", admin.id, "requestId", request, "expectedRevision", target.path("revision").asLong(),
                "expectedFactsHash", target.path("factsHash").asText(),
                "note", "受控测试：已核实原供应商调用结束，仅清理执行状态", "providerFinished", true);
    }
    private JsonNode recover(Fixture admin, String kind, String id) {
        String path = "/api/system/execution-recovery/" + kind + "/" + id;
        return ok(post(admin, path + "/confirm-finished", recoveryDecision(admin, ok(get(admin, path)), id())));
    }

    private record Fixture(long id, String canvas, HttpHeaders headers) { }
    private Fixture fixture() {
        long id = IDS.incrementAndGet();
        jdbc.update("INSERT INTO sys_user(id, username, password) VALUES (?, ?, 'test-only')", id, "canvas-run-" + id);
        jdbc.update("INSERT INTO afv_project(id, name, owner_type, owner_id) VALUES (?, '运行测试项目', 1, ?)", id, id);
        if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config", Integer.class) == 0)
            jdbc.update("INSERT INTO afv_storage_config(name, type, base_path, is_default) VALUES ('运行测试存储', 'local', ?, TRUE)", FILES.toString());
        var headers = new HttpHeaders();
        headers.setBearerAuth(tokens.createToken(id, "canvas-run-" + id, null).getAccessToken());
        headers.setContentType(MediaType.APPLICATION_JSON);
        String canvas = id();
        var fixture = new Fixture(id, canvas, headers);
        ok(post(fixture, "/api/canvases", Map.of("id", canvas, "title", "运行测试画布", "projectId", id)));
        return fixture;
    }
    private void model(Fixture f) {
        jdbc.update("INSERT INTO afv_api_config(id, name, platform, image_protocol, api_url, api_key) VALUES (?, '模拟图片接口', 'openai_compatible', 'openai', ?, 'test-key')",
                f.id, "http://localhost:" + PROVIDER.getAddress().getPort());
        jdbc.update("""
                INSERT INTO afv_ai_model(id, name, code, model_type, model_protocol, api_config_id, config, max_concurrency,
                    multimodal_input_types, multimodal_input_transports, reasoning_effort_levels)
                VALUES (?, '模拟图片模型', 'gpt-image-2', 2, 'openai', ?, '{"defaultWidth":32,"defaultHeight":32}', 1, JSON_ARRAY(), JSON_OBJECT(), JSON_ARRAY())
                """, f.id, f.id);
    }
    private void audioModel(Fixture f) {
        jdbc.update("INSERT INTO afv_api_config(id, name, platform, api_url, api_key) VALUES (?, '模拟音频接口', 'sonilo', ?, 'test-key')",
                f.id, "http://localhost:" + PROVIDER.getAddress().getPort());
    }
    private TextScenario textModel(Fixture f, int status, boolean wait, String finish) {
        jdbc.update("INSERT INTO afv_api_config(id, name, platform, text_protocol, api_url, api_key) VALUES (?, '模拟文本接口', 'openai_compatible', 'openai_compatible', ?, 'test-key')",
                f.id, "http://localhost:" + PROVIDER.getAddress().getPort());
        jdbc.update("""
                INSERT INTO afv_ai_model(id, name, code, model_type, model_protocol, api_config_id, config, max_concurrency,
                    multimodal_input_types, multimodal_input_transports, reasoning_effort_levels)
                VALUES (?, '模拟文本模型', 'test-text-model', 1, 'openai_compatible', ?, '{"maxTokens":128}', 1, JSON_ARRAY(), JSON_OBJECT(), JSON_ARRAY())
                """, f.id, f.id);
        var scenario = new TextScenario(status, wait, finish);
        TEXT.put("text-" + f.id, scenario);
        return scenario;
    }
    private String upload(Fixture f, byte[] bytes, String name, String mime) {
        var values = new LinkedMultiValueMap<String, Object>();
        var part = new HttpHeaders(); part.setContentType(MediaType.parseMediaType(mime));
        values.add("file", new HttpEntity<>(new ByteArrayResource(bytes) { @Override public String getFilename() { return name; } }, part));
        values.add("requestId", id()); values.add("projectId", String.valueOf(f.id)); values.add("canvasId", f.canvas);
        var headers = new HttpHeaders(); headers.addAll(f.headers); headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        return ok(http.postForEntity("/api/media-resources/upload", new HttpEntity<>(values, headers), JsonNode.class)).path("id").asText();
    }
    private byte[] video(Fixture f, String color, boolean audio) throws Exception {
        Files.createDirectories(FILES);
        Path video = FILES.resolve("test-" + f.id + "-" + color + ".mp4");
        Path log = FILES.resolve("test-" + f.id + "-" + color + ".log");
        List<String> args = new ArrayList<>(List.of("ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=" + color + ":s=64x48:r=12"));
        if (audio) args.addAll(List.of("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"));
        args.addAll(List.of("-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", video.toString()));
        var process = new ProcessBuilder(args).redirectErrorStream(true).redirectOutput(log.toFile()).start();
        if (!process.waitFor(15, TimeUnit.SECONDS)) { process.destroyForcibly(); throw new IOException("Fixture video encoding timed out"); }
        assertThat(process.exitValue()).as(Files.readString(log)).isZero();
        return Files.readAllBytes(video);
    }
    private static byte[] wav() {
        int length = 48000 * 3 * 2;
        var buffer = ByteBuffer.allocate(44 + length).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + length).put("WAVEfmt ".getBytes(StandardCharsets.US_ASCII))
                .putInt(16).putShort((short) 1).putShort((short) 1).putInt(48000).putInt(96000).putShort((short) 2).putShort((short) 16)
                .put("data".getBytes(StandardCharsets.US_ASCII)).putInt(length);
        for (int i = 0; i < length / 2; i++) buffer.putShort((short) (Math.sin(i * 440 * 2 * Math.PI / 48000) * 2000));
        return buffer.array();
    }
    private static final class AudioScenario {
        final AtomicInteger posts = new AtomicInteger();
        final AtomicBoolean complete = new AtomicBoolean();
        final boolean uncertain;
        volatile boolean rejected;
        AudioScenario(boolean uncertain) { this.uncertain = uncertain; }
    }
    private static final class TextScenario {
        final AtomicInteger posts = new AtomicInteger();
        final CountDownLatch entered = new CountDownLatch(1);
        final CountDownLatch release;
        volatile int status;
        final String finish;
        volatile JsonNode request;
        TextScenario(int status, boolean wait, String finish) {
            this.status = status;
            this.release = new CountDownLatch(wait ? 1 : 0);
            this.finish = finish;
        }
    }
    private Map<String, Object> node(String id, String kind, Map<String, Object> config) {
        return Map.of("id", id, "kind", kind, "title", kind, "schemaVersion", 1, "config", config, "configVersion", 0, "layoutVersion", 0,
                "layout", Map.of("x", 40, "y", 80, "width", 300, "locked", false, "collapsed", false));
    }
    private Map<String, Object> batch(String operation, Map<String, Object> config, String input, String output, String inputType, String resultType, int max) {
        return Map.of("operationKind", operation, "operationConfig", config, "inputPort", input, "outputPort", output,
                "inputType", inputType, "resultType", resultType, "maxItems", max);
    }
    private Map<String, Object> resourceNode(String id, String type, long resourceId) {
        var result = new HashMap<>(node(id, type, Map.of()));
        result.put("resource", Map.of("type", type, "id", resourceId, "version", 0));
        return result;
    }
    private Map<String, Object> writeback(String target, String value, String field) {
        return Map.of("binding", target + ":" + field, "targetType", target, "valueType", value);
    }
    private Map<String, Object> decision(Fixture f, JsonNode node, String action) {
        return Map.of("expectedRevision", node.path("revision").asLong(), "proposalHash", node.at("/writeback/hash").asText(),
                "action", action, "note", "已核对本次具体改动及版本", "scope", scope(f));
    }
    private List<JsonNode> children(JsonNode view, String parent) {
        List<JsonNode> result = new ArrayList<>();
        view.path("nodes").forEach(node -> { if (node.path("parentNodeRunId").asText().equals(parent)) result.add(node); });
        return result;
    }
    private Map<String, Object> edge(String source, String sourcePort, String target, String targetPort, int index) {
        return Map.of("id", id(), "kind", "execution", "sourceNodeId", source, "sourcePort", sourcePort,
                "targetNodeId", target, "targetPort", targetPort, "inputIndex", index, "version", 0);
    }
    private void create(Fixture f, List<Map<String, Object>> nodes, List<Map<String, Object>> edges) {
        List<Object> ops = new ArrayList<>();
        nodes.forEach(node -> ops.add(Map.of("type", "canvas.node.create", "node", node)));
        edges.forEach(edge -> ops.add(Map.of("type", "canvas.edge.create", "edge", edge)));
        commands(f, ops);
    }
    private void commands(Fixture f, List<?> ops) { ok(post(f, "/api/canvases/" + f.canvas + "/commands", Map.of("requestId", id(), "operations", ops))); }
    private JsonNode snapshot(Fixture f) { return ok(get(f, "/api/canvases/" + f.canvas)); }
    private Map<String, Object> scope(Fixture f) { return Map.of("userId", f.id); }
    private JsonNode plan(Fixture f, String mode, List<String> ids, Map<String, String> reuse) {
        return ok(post(f, "/api/canvases/" + f.canvas + "/run-plans", Map.of("expectedRevision", snapshot(f).at("/document/revision").asLong(),
                "mode", mode, "nodeIds", ids, "reuse", reuse, "scope", scope(f),
                "budget", Map.of("maxNodes", 100, "maxImages", 32, "maxVideoSeconds", 120, "maxParallel", 2, "maxBatchItems", 100))));
    }
    private Map<String, Object> startRequest(Fixture f, JsonNode plan) {
        return Map.of("requestId", id(), "planId", plan.path("id").asText(), "planHash", plan.path("hash").asText(), "scope", scope(f));
    }
    private String start(Fixture f, JsonNode plan) {
        assertThat(plan.path("issues")).as(plan.toString()).isEmpty();
        return ok(post(f, "/api/canvases/" + f.canvas + "/runs", startRequest(f, plan))).path("id").asText();
    }
    private JsonNode awaitState(Fixture f, String id, String state) {
        await().pollInterval(Duration.ofMillis(200)).atMost(Duration.ofSeconds(35)).untilAsserted(() -> {
            var result = ok(get(f, "/api/canvas-runs/" + id));
            var errors = new ArrayList<String>();
            result.path("nodes").forEach(node -> errors.add(node.path("kind").asText() + ": " + node.path("error").asText()));
            assertThat(result.path("status").asText()).as("Node errors: %s", errors).isEqualTo(state);
        });
        return ok(get(f, "/api/canvas-runs/" + id));
    }
    private JsonNode nodeResult(JsonNode run, String nodeId) {
        for (var node : run.path("nodes")) if (node.path("nodeId").asText().equals(nodeId)) return node;
        throw new AssertionError("Missing node " + nodeId);
    }
    private ResponseEntity<JsonNode> post(Fixture f, String path, Object body) { return http.postForEntity(path, new HttpEntity<>(body, f.headers), JsonNode.class); }
    private ResponseEntity<JsonNode> get(Fixture f, String path) { return http.exchange(path, HttpMethod.GET, new HttpEntity<>(f.headers), JsonNode.class); }
    private JsonNode ok(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().path("code").asInt()).as(response.toString()).isZero();
        return response.getBody().path("data");
    }
    private static String id() { return UUID.randomUUID().toString(); }
    private static HttpServer provider() {
        try {
            var server = HttpServer.create(new InetSocketAddress(0), 0);
            server.createContext("/v1/chat/completions", exchange -> {
                var request = JSON.readTree(exchange.getRequestBody());
                var scenario = TEXT.entrySet().stream().filter(entry -> request.toString().contains(entry.getKey())).findFirst().orElseThrow().getValue();
                scenario.request = request;
                scenario.posts.incrementAndGet();
                scenario.entered.countDown();
                try {
                    if (!scenario.release.await(25, TimeUnit.SECONDS)) throw new IOException("测试文本响应未解除等待");
                } catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IOException(error); }
                if (scenario.status == 0) { exchange.close(); return; }
                boolean streaming = scenario.status == 200 && request.path("stream").asBoolean();
                String responseJson = JSON.writeValueAsString(scenario.status == 200 ? Map.of("id", "response-test", "choices", List.of(Map.of(
                        "index", 0, streaming ? "delta" : "message", Map.of("role", "assistant", "content", "模型返回文本"), "finish_reason", scenario.finish)),
                        "usage", Map.of("prompt_tokens", 10, "completion_tokens", 5, "total_tokens", 15))
                        : Map.of("error", Map.of("message", "text queue is full")));
                byte[] response = (streaming ? "data: " + responseJson + "\n\ndata: [DONE]\n\n" : responseJson).getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", streaming ? "text/event-stream" : "application/json");
                exchange.sendResponseHeaders(scenario.status, response.length);
                try (var out = exchange.getResponseBody()) { out.write(response); }
            });
            server.createContext("/v1/images/generations", exchange -> {
                var request = JSON.readTree(exchange.getRequestBody());
                POSTS.computeIfAbsent(request.path("prompt").asText(), ignored -> new AtomicInteger()).incrementAndGet();
                var outputs = JSON.createArrayNode();
                for (int index = 0; index < request.path("n").asInt(1); index++) outputs.addObject().put("b64_json", Base64.getEncoder().encodeToString(PNG));
                byte[] response = JSON.writeValueAsBytes(Map.of("data", outputs));
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                if (request.path("prompt").asText().startsWith("recovery-unknown-")) { exchange.close(); return; }
                exchange.sendResponseHeaders(200, response.length);
                try (var out = exchange.getResponseBody()) { out.write(response); }
            });
            server.createContext("/v1/images/edits", exchange -> {
                byte[] bytes = exchange.getRequestBody().readAllBytes();
                String body = new String(bytes, StandardCharsets.ISO_8859_1);
                String prompt = IMAGE_EDIT_OUTPUTS.keySet().stream().filter(body::contains).findFirst().orElseThrow();
                IMAGE_EDIT_REQUESTS.put(prompt, bytes);
                POSTS.computeIfAbsent(prompt, ignored -> new AtomicInteger()).incrementAndGet();
                byte[] response = JSON.writeValueAsBytes(Map.of("data", List.of(Map.of("b64_json",
                        Base64.getEncoder().encodeToString(IMAGE_EDIT_OUTPUTS.get(prompt))))));
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, response.length);
                try (var out = exchange.getResponseBody()) { out.write(response); }
            });
            for (String operation : List.of("music", "sfx")) server.createContext("/v1/video-to-" + operation, exchange -> {
                String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.ISO_8859_1);
                var entry = AUDIO.entrySet().stream().filter(item -> body.contains(item.getKey())).findFirst().orElseThrow();
                entry.getValue().posts.incrementAndGet();
                if (entry.getValue().uncertain) { exchange.close(); return; }
                byte[] response = JSON.writeValueAsBytes(entry.getValue().rejected
                        ? Map.of("error", "audio queue is full") : Map.of("task_id", entry.getKey()));
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(entry.getValue().rejected ? 503 : 200, response.length);
                try (var out = exchange.getResponseBody()) { out.write(response); }
            });
            server.createContext("/v1/tasks/", exchange -> {
                String id = exchange.getRequestURI().getPath().substring("/v1/tasks/".length());
                var task = AUDIO.get(id);
                byte[] response = JSON.writeValueAsBytes(Map.of("status", task.complete.get() ? "succeeded" : "processing",
                        "audio", Map.of("url", "data:audio/wav;base64," + Base64.getEncoder().encodeToString(wav()))));
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, response.length);
                try (var out = exchange.getResponseBody()) { out.write(response); }
            });
            server.start();
            return server;
        } catch (IOException failure) { throw new IllegalStateException(failure); }
    }
    private static byte[] png() {
        try {
            var out = new ByteArrayOutputStream();
            ImageIO.write(new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB), "png", out);
            return out.toByteArray();
        } catch (IOException failure) { throw new IllegalStateException(failure); }
    }
}
