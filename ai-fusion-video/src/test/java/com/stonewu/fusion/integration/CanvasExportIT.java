package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportData;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportCleanup;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportRepository;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportTransactions;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportWorker;
import com.stonewu.fusion.service.canvas.delivery.CanvasExportWriter;
import com.stonewu.fusion.service.media.MediaBlobStore;
import com.stonewu.fusion.service.media.MediaSourceImportService;
import com.stonewu.fusion.service.media.MediaCopyService;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.storage.LegacyMediaRoot;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.util.LinkedMultiValueMap;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Files;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import java.util.zip.ZipInputStream;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.awaitility.Awaitility.await;

/** HTTP + 真实数据库、Redis、加密文件、下载与运行调度；仅使用临时容器。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, properties = {"app.canvas.exports.enabled=false","app.media.cleanup.initial-delay-ms=3600000"})
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class CanvasExportIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test").withCommand("--default-time-zone=+08:00");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "canvas-export-" + UUID.randomUUID()).toAbsolutePath();
    static final AtomicLong IDS = new AtomicLong(85000);
    @DynamicPropertySource static void properties(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url", MYSQL::getJdbcUrl);
        properties.add("spring.datasource.username", MYSQL::getUsername);
        properties.add("spring.datasource.password", MYSQL::getPassword);
        properties.add("spring.data.redis.host", REDIS::getHost);
        properties.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379));
        properties.add("spring.data.redis.password", () -> "");
        properties.add("app.storage.local-base-path", FILES::toString);
        properties.add("app.agent-workspace.local-base-path", () -> FILES.resolve("workspace").toString());
    }
    @Autowired TestRestTemplate http;
    @Autowired TokenService tokens;
    @Autowired JdbcTemplate jdbc;
    @Autowired MediaSourceImportService sourceImports;
    @Autowired LegacyMediaRoot legacyRoot;
    @Autowired ResourceQueryService resourceQueries;
    @Autowired MediaCopyService mediaCopies;
    @Autowired CanvasJson json;
    @Autowired PlatformTransactionManager transactionManager;
    @Autowired CanvasExportRepository jobs;
    @Autowired CanvasExportTransactions transactions;
    @Autowired CanvasExportWriter writer;
    @Autowired MediaBlobStore blobs;
    record Fixture(long id, String canvas, HttpHeaders owner, HttpHeaders other) { }

    @Test void frozenJsonIsIdempotentEncryptedAndSupportsAuthorizedRangeDownloads() {
        var f = fixture(false);
        create(f, List.of(node("text", Map.of("content", "交付前的文字"), null)), List.of());
        var options = options(f, "JSON", "CANVAS");
        var preview = ok(post(f, "/exports/preview", options));
        var body = Map.of("requestId", UUID.randomUUID().toString(), "options", options, "previewHash", preview.path("hash").asText());
        var requests = new ArrayList<CompletableFuture<JsonNode>>();
        for (int i = 0; i < 5; i++) requests.add(CompletableFuture.supplyAsync(() -> ok(post(f, "/exports", body))));
        String job = requests.getFirst().join().path("id").asText();
        assertThat(requests.stream().map(value -> value.join().path("id").asText()).distinct()).containsExactly(job);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_export WHERE canvas_id=?", Integer.class, f.canvas())).isEqualTo(1);
        create(f, List.of(node("text", Map.of("content", "提交后的新文字"), null)), List.of());
        build(job);
        byte[] download = download(f, job);
        String text = new String(download, StandardCharsets.UTF_8);
        assertThat(text).contains("交付前的文字").doesNotContain("提交后的新文字", "encryptionKeyset", "keyset", "apiKey", "ownerInstanceId");
        var stored = jobs.get(job, false);
        assertThat(stored.status()).isEqualTo("SUCCEEDED");
        assertThat(stored.createdAt()).isBetween(Instant.now().minusSeconds(60), Instant.now());
        var blob = jobs.blob(stored.blobId());
        assertThat(blob.storedBytes()).isGreaterThan(stored.byteSize());
        var access = ok(post(f, "/exports/" + job + "/access", Map.of()));
        var headers = new HttpHeaders(); headers.set(HttpHeaders.RANGE, "bytes=0-31");
        var range = http.exchange(access.path("url").asText(), HttpMethod.GET, new HttpEntity<>(headers), byte[].class);
        assertThat(range.getStatusCode()).isEqualTo(HttpStatus.PARTIAL_CONTENT);
        assertThat(range.getBody()).isEqualTo(Arrays.copyOf(download, 32));
        assertThat(http.getForEntity("/api/canvas-export-content/" + job + "?ticket=wrong", String.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        jdbc.update("UPDATE sys_user SET status=0 WHERE id=?", f.id());
        assertThat(http.getForEntity(access.path("url").asText(), String.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void staleBusinessPreviewCannotExportNewContentWithoutConfirmation() {
        var f = fixture(true);
        create(f, List.of(node("script", Map.of(), Map.of("type", "script", "id", f.id(), "version", 0))), List.of());
        var options = options(f, "MARKDOWN", "CANVAS");
        var preview = ok(post(f, "/exports/preview", options));
        ok(post(f, "/resource-commands", Map.of("requestId", UUID.randomUUID().toString(), "resourceType", "script", "resourceId", f.id(),
                "expectedVersion", 0, "patch", Map.of("summary", "其他编辑者的新内容"))));
        assertThat(post(f, "/exports", Map.of("requestId", UUID.randomUUID().toString(), "options", options, "previewHash", preview.path("hash").asText())).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM afv_resource_event WHERE project_id=? AND published=FALSE",Integer.class,f.id())).isZero());
        String job = submit(f, "MARKDOWN", "CANVAS"); build(job);
        assertThat(new String(download(f, job), StandardCharsets.UTF_8)).contains("其他编辑者的新内容", "版本 1");
    }

    @Test void zipContainsRealMediaAndSharingDoesNotGivePrivateFileAccess() throws Exception {
        var f = fixture(false);
        byte[] original = "只属于作者的素材\n".getBytes(StandardCharsets.UTF_8);
        String media = upload(f, original);
        create(f, List.of(node("textFile", Map.of("resourceId", media), null)), List.of());
        String job = submit(f, "ZIP", "CANVAS"); build(job);
        var contents = new LinkedHashMap<String, byte[]>();
        try (var zip = new ZipInputStream(new ByteArrayInputStream(download(f, job)), StandardCharsets.UTF_8)) {
            for (var entry = zip.getNextEntry(); entry != null; entry = zip.getNextEntry()) contents.put(entry.getName(), zip.readAllBytes());
        }
        assertThat(contents.keySet()).contains("manifest.json", "script.md", "storyboard.csv", "README.txt");
        assertThat(contents.entrySet().stream().filter(entry -> entry.getKey().startsWith("media/")).map(Map.Entry::getValue)).containsExactly(original);
        ok(post(f, "/members", Map.of("requestId", UUID.randomUUID().toString(), "expectedRevision", revision(f), "userId", f.id()+1,
                "canEdit", false, "canExecute", false, "canManage", false, "remove", false)));
        assertThat(http.postForEntity(base(f)+"/exports/"+job+"/access", new HttpEntity<>(Map.of(), f.other()), JsonNode.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var listed = ok(http.exchange(base(f)+"/exports", HttpMethod.GET, new HttpEntity<>(f.other()), JsonNode.class));
        assertThat(listed.get(0).path("contentUnavailable").asBoolean()).isTrue();
        assertThat(listed.toString()).doesNotContain("只属于作者", "私有.txt", media);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference_document WHERE anchor_kind='CANVAS_EXPORT' AND anchor_id=?", Integer.class, job)).isEqualTo(1);
    }

    @Test void singleMediaExportReturnsTheOriginalBytes() {
        var f = fixture(false);
        byte[] original = "原文件内容".getBytes(StandardCharsets.UTF_8);
        String media = upload(f, original);
        create(f, List.of(node("textFile", Map.of("resourceId", media), null)), List.of());
        String job = submit(f, "MEDIA", "CANVAS"); build(job);
        assertThat(download(f, job)).isEqualTo(original);
        assertThat(jobs.get(job, false).mimeType()).startsWith("text/plain");
    }

    @Test void expiredWorkerCannotPublishAndNewWorkerResumesTheSameSnapshot() throws Exception {
        var f = fixture(false);
        create(f, List.of(node("text", Map.of("content", "重启后保留"), null)), List.of());
        String job = submit(f, "JSON", "CANVAS");
        var first = transactions.claim(job);
        var abandoned = transactions.reserve(first);
        Files.createDirectories(FILES);
        Path incomplete=Files.createTempFile(FILES,"abandoned-export-",".json");
        Files.writeString(incomplete,"{\"unfinished\":true}");
        blobs.writeDelivery(abandoned,incomplete);
        jdbc.update("UPDATE afv_canvas_export SET lease_until=TIMESTAMPADD(SECOND,-1,UTC_TIMESTAMP(6)) WHERE id=?", job);
        var next = transactions.claim(job);
        assertThat(next.epoch()).isGreaterThan(first.epoch());
        assertThatThrownBy(() -> transactions.complete(first, abandoned, 10, 1, "0".repeat(64))).isInstanceOf(BusinessException.class);
        assertThat(jdbc.queryForObject("SELECT state FROM afv_canvas_export_blob WHERE id=?", String.class, abandoned.id())).isEqualTo("ORPHANED");
        worker().execute(next);
        assertThat(jobs.get(job, false).status()).isEqualTo("SUCCEEDED");
        assertThat(jobs.get(job, false).attempts()).isEqualTo(2);
        var cleanup=new CanvasExportCleanup(transactions,blobs,30);
        assertThat(cleanup.runOne()).isFalse();
        byte[] published=download(f,job);
        jdbc.update("UPDATE afv_canvas_export_blob SET update_time=TIMESTAMPADD(DAY,-31,UTC_TIMESTAMP(6)) WHERE id=?",abandoned.id());
        assertThat(cleanup.runOne()).isTrue();
        assertThat(FILES.resolve(".afv-managed").resolve(abandoned.objectKey())).doesNotExist();
        assertThat(jdbc.queryForObject("SELECT state FROM afv_canvas_export_blob WHERE id=?",String.class,abandoned.id())).isEqualTo("PURGED");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_export_event WHERE export_id=? AND event_type='FILE_PURGED'",Long.class,job)).isEqualTo(1);
        assertThatThrownBy(() -> blobs.writeDelivery(abandoned,incomplete)).isInstanceOf(IOException.class);
        assertThat(download(f,job)).isEqualTo(published);
    }

    @Test void cancellationAndRevocationPreventArtifactPublication() {
        var f = fixture(false);
        String queued = submit(f, "JSON", "CANVAS");
        ok(post(f, "/exports/"+queued+"/cancel", Map.of()));
        assertThat(transactions.claim(queued)).isNull();
        assertThat(jobs.get(queued, false).status()).isEqualTo("CANCELLED");
        String running = submit(f, "JSON", "CANVAS");
        var claim = transactions.claim(running);
        ok(post(f, "/exports/"+running+"/cancel", Map.of()));
        worker().execute(claim);
        assertThat(jobs.get(running, false).status()).isEqualTo("CANCELLED");
        assertThat(jobs.get(running, false).blobId()).isNull();
        String revoked = submit(f, "JSON", "CANVAS");
        var original = transactions.claim(revoked);
        jdbc.update("UPDATE sys_user SET status=0 WHERE id=?", f.id());
        worker().execute(original);
        assertThat(jobs.get(revoked, false).status()).isEqualTo("FAILED");
        assertThat(jobs.get(revoked, false).blobId()).isNull();
    }

    @Test void exportNodeUsesTheDurableJobAndPublishesDownloadReference() {
        var f = fixture(false);
        var text = node("text", Map.of("content", "节点导出结果"), null);
        var output = node("export", Map.of("valueType", "text", "format", "json"), null);
        var edge = Map.of("id", UUID.randomUUID().toString(), "kind", "execution", "sourceNodeId", text.get("id"), "sourcePort", "text", "targetNodeId", output.get("id"), "targetPort", "items", "inputIndex", 0, "version", 0);
        create(f, List.of(text, output), List.of(edge));
        var scope = Map.of("userId", f.id());
        var plan = ok(post(f, "/run-plans", Map.of("expectedRevision", revision(f), "mode", "ALL", "nodeIds", List.of(), "reuse", Map.of(), "scope", scope,
                "budget", Map.of("maxNodes", 10, "maxImages", 0, "maxVideoSeconds", 0, "maxParallel", 2, "maxBatchItems", 10))));
        assertThat(plan.path("issues")).isEmpty();
        var run = ok(post(f, "/runs", Map.of("requestId", UUID.randomUUID().toString(), "planId", plan.path("id").asText(), "planHash", plan.path("hash").asText(), "scope", scope)));
        await().atMost(Duration.ofSeconds(15)).until(() -> jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_export WHERE canvas_id=?", Integer.class, f.canvas()) == 1);
        String job = jdbc.queryForObject("SELECT id FROM afv_canvas_export WHERE canvas_id=?", String.class, f.canvas());
        build(job);
        await().atMost(Duration.ofSeconds(15)).until(() -> jdbc.queryForObject("SELECT status FROM afv_canvas_run WHERE id=?", String.class, run.path("id").asText()).equals("SUCCEEDED"));
        assertThat(jdbc.queryForObject("SELECT external_kind FROM afv_canvas_node_run WHERE id=?", String.class, jobs.get(job, false).nodeRunId())).isEqualTo("EXPORT");
        assertThat(new String(download(f, job), StandardCharsets.UTF_8)).contains("节点导出结果", "sourceNodeRunId");
    }

    // 处理后的单个结果不能被上游素材扩大成多个交付文件，来源失效后仍须拒绝下载。
    @ParameterizedTest
    @ValueSource(strings = {"media", "zip"})
    void exportNodeDeliversOnlyItsInputAndStillChecksSourceMedia(String format) throws Exception {
        var f = fixture(false);
        var imageBytes = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB), "png", imageBytes);
        String original = upload(f, imageBytes.toByteArray(), "原始图片.png");
        var image = node("image", Map.of("resourceId", original), null);
        var resize = node("imageTransform", Map.of("operation", "resize", "width", 2, "height", 2, "fit", "fill"), null);
        var output = node("export", Map.of("valueType", "image", "format", format), null);
        create(f, List.of(image, resize, output), List.of(
                edge(image, "media", resize, "image"), edge(resize, "image", output, "items")));
        var scope = Map.of("userId", f.id());
        var plan = ok(post(f, "/run-plans", Map.of("expectedRevision", revision(f), "mode", "ALL",
                "nodeIds", List.of(), "reuse", Map.of(), "scope", scope,
                "budget", Map.of("maxNodes", 10, "maxImages", 0, "maxVideoSeconds", 0,
                        "maxParallel", 2, "maxBatchItems", 10))));
        assertThat(plan.path("issues")).isEmpty();
        var run = ok(post(f, "/runs", Map.of("requestId", UUID.randomUUID().toString(),
                "planId", plan.path("id").asText(), "planHash", plan.path("hash").asText(), "scope", scope)));
        await().atMost(Duration.ofSeconds(30)).until(() -> jdbc.queryForObject(
                "SELECT COUNT(*) FROM afv_canvas_export WHERE canvas_id=?", Integer.class, f.canvas()) == 1);
        String job = jdbc.queryForObject("SELECT id FROM afv_canvas_export WHERE canvas_id=?", String.class, f.canvas());
        build(job);
        await().atMost(Duration.ofSeconds(15)).until(() -> "SUCCEEDED".equals(jdbc.queryForObject(
                "SELECT status FROM afv_canvas_run WHERE id=?", String.class, run.path("id").asText())));
        var manifest = jobs.get(job, false).manifest();
        assertThat(manifest.media()).hasSize(1);
        assertThat(manifest.media().getFirst().id()).isNotEqualTo(original);
        assertThat(manifest.graph().nodes()).hasSize(3);
        byte[] delivered = download(f, job);
        if (format.equals("zip")) {
            var packagedMedia = new ArrayList<byte[]>();
            try (var zip = new ZipInputStream(new ByteArrayInputStream(delivered), StandardCharsets.UTF_8)) {
                for (var entry = zip.getNextEntry(); entry != null; entry = zip.getNextEntry()) {
                    if (entry.getName().startsWith("media/")) {
                        packagedMedia.add(zip.readAllBytes());
                    }
                }
            }
            assertThat(packagedMedia).hasSize(1);
            delivered = packagedMedia.getFirst();
        }
        BufferedImage result = ImageIO.read(new ByteArrayInputStream(delivered));
        assertThat(result.getWidth()).isEqualTo(2);
        assertThat(result.getHeight()).isEqualTo(2);
        jdbc.update("UPDATE afv_media_resource SET state='DELETED',version=version+1 WHERE id=?", original);
        var unavailable = post(f, "/exports/" + job + "/access", Map.of());
        assertThat(unavailable.getStatusCode()).isEqualTo(HttpStatus.GONE);
        assertThat(unavailable.getBody().path("msg").asText()).isEqualTo("媒体资源已删除");
    }

    @Test void wholeProjectCsvIncludesUnplacedShotsAndEscapesFormulaAndQuotedLines() {
        var f = fixture(true);
        jdbc.update("INSERT INTO afv_storyboard(id,project_id,script_id,title) VALUES(?,?,?,'交付分镜')",f.id(),f.id(),f.id());
        jdbc.update("INSERT INTO afv_storyboard_episode(id,storyboard_id,title) VALUES(?,?,'第一集')",f.id(),f.id());
        jdbc.update("INSERT INTO afv_storyboard_scene(id,storyboard_id,episode_id) VALUES(?,?,?)",f.id(),f.id(),f.id());
        jdbc.update("INSERT INTO afv_storyboard_item(id,storyboard_id,storyboard_episode_id,storyboard_scene_id,content,dialogue,sort_order) VALUES(?,?,?,?,?,?,0)",
                f.id(),f.id(),f.id(),f.id(),"=SUM(1,2)","对白含\"引号\"\n第二行");
        jdbc.update("INSERT INTO afv_storyboard_item(id,storyboard_id,storyboard_episode_id,content,sort_order) VALUES(?,?,?,'尚未归入场次的镜头',1)", f.id()+1,f.id(),f.id());
        jdbc.update("INSERT INTO afv_storyboard_item(id,storyboard_id,content,sort_order) VALUES(?,?,'尚未归入分集的镜头',2)", f.id()+2,f.id());
        String job = submit(f,"CSV","PROJECT"); build(job);
        String csv = new String(download(f,job),StandardCharsets.UTF_8);
        assertThat(csv).startsWith("\uFEFF").contains("\"'=SUM(1,2)\"", "\"对白含\"\"引号\"\"\n第二行\"", "[version]");
        assertThat(csv).contains("尚未归入场次的镜头", "尚未归入分集的镜头");
        assertThat(jobs.get(job,false).manifest().values()).hasSize(8);
    }

    @Test void selectionExportsOnlyChosenNodesAndRequiresExactScopeAndFormat() {
        var f = fixture(false);
        var first = node("text",Map.of("content","选中的内容"),null);
        var second = node("text",Map.of("content","范围外的内容"),null);
        create(f,List.of(first,second),List.of());
        var options = Map.of("format","JSON","scope","SELECTION","nodeIds",List.of(first.get("id")),"expectedRevision",revision(f));
        var preview = ok(post(f,"/exports/preview",options));
        String job = ok(post(f,"/exports",Map.of("requestId",UUID.randomUUID().toString(),"options",options,"previewHash",preview.path("hash").asText()))).path("id").asText();
        build(job);
        assertThat(new String(download(f,job),StandardCharsets.UTF_8)).contains("选中的内容").doesNotContain("范围外的内容");
        assertThat(post(f,"/exports/preview",options(f,"JSON","PROJECT")).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(post(f,"/exports/preview",options(f,"MEDIA","CANVAS")).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(post(f,"/exports/preview",options(f,"CSV","CANVAS")).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }

    private Fixture fixture(boolean project) {
        long id = IDS.addAndGet(1000);
        jdbc.update("INSERT INTO sys_user(id,username,password) VALUES(?,?,'test'),(?,?,'test')", id, "export-"+id, id+1, "export-"+(id+1));
        if (project) { jdbc.update("INSERT INTO afv_project(id,name,owner_type,owner_id) VALUES(?,'交付项目',1,?)", id,id); jdbc.update("INSERT INTO afv_script(id,project_id,title,summary) VALUES(?,?,'交付剧本','原来的概要')",id,id); }
        if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config WHERE is_default=TRUE AND deleted=FALSE", Integer.class) == 0)
            jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('交付测试存储','local',?,TRUE)", FILES.toString());
        else jdbc.update("UPDATE afv_storage_config SET base_path=? WHERE is_default=TRUE AND deleted=FALSE", FILES.toString());
        var body = new HashMap<String,Object>(); body.put("id", UUID.randomUUID().toString()); body.put("title", "交付验收"); body.put("projectId", project ? id : null);
        var f = new Fixture(id, body.get("id").toString(), headers(id), headers(id+1));
        ok(http.postForEntity("/api/canvases", new HttpEntity<>(body,f.owner()), JsonNode.class)); return f;
    }

    @Test void activityAggregatesRealRunsAndDeliveriesAndUsesExclusiveEndDates() {
        var f = fixture(false);
        var text = node("text", Map.of("content", "统计真实执行"), null);
        var output = node("export", Map.of("valueType", "text", "format", "json"), null);
        create(f, List.of(text, output), List.of(Map.of("id", UUID.randomUUID().toString(), "kind", "execution",
                "sourceNodeId", text.get("id"), "sourcePort", "text", "targetNodeId", output.get("id"), "targetPort", "items", "inputIndex", 0, "version", 0)));
        Instant from = Instant.now().minusSeconds(1);
        String firstCreatedAt = null;
        for (int i=0; i<2; i++) {
            var scope = Map.of("userId", f.id());
            var plan = ok(post(f, "/run-plans", Map.of("expectedRevision", revision(f), "mode", "ALL", "nodeIds", List.of(), "reuse", Map.of(), "scope", scope,
                    "budget", Map.of("maxNodes", 10, "maxImages", 0, "maxVideoSeconds", 0, "maxParallel", 2, "maxBatchItems", 10))));
            assertThat(plan.path("issues")).isEmpty();
            var run = ok(post(f, "/runs", Map.of("requestId", UUID.randomUUID().toString(), "planId", plan.path("id").asText(), "planHash", plan.path("hash").asText(), "scope", scope)));
            if (firstCreatedAt == null) firstCreatedAt = run.path("createdAt").asText();
            String runId = run.path("id").asText();
            await().atMost(Duration.ofSeconds(15)).until(() -> jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_export e JOIN afv_canvas_node_run n ON n.id=e.node_run_id WHERE n.run_id=?",Integer.class,runId)==1);
            build(jdbc.queryForObject("SELECT e.id FROM afv_canvas_export e JOIN afv_canvas_node_run n ON n.id=e.node_run_id WHERE n.run_id=?",String.class,runId));
            await().atMost(Duration.ofSeconds(15)).until(() -> jdbc.queryForObject("SELECT status FROM afv_canvas_run WHERE id=?",String.class,runId).equals("SUCCEEDED"));
        }
        var result = ok(activity(f, from, Instant.now().plusSeconds(1), f.owner()));
        assertThat(result.path("runs").asInt()).isEqualTo(2);
        assertThat(result.at("/states/SUCCEEDED").asInt()).isEqualTo(2);
        assertThat(result.at("/nodes/total").asInt()).isEqualTo(4);
        assertThat(result.at("/nodes/succeeded").asInt()).isEqualTo(4);
        assertThat(result.at("/tokens/total").isNull()).isTrue();
        assertThat(result.at("/deliveries/total").asInt()).isEqualTo(2);
        assertThat(result.at("/deliveries/bytes").asLong()).isPositive();
        assertThat(ok(activity(f,from,Instant.parse(firstCreatedAt),f.owner())).path("runs").asInt()).isZero();
        assertThat(activity(f,from,Instant.now().plusSeconds(1),f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void activityDoesNotExposePrivateDeliverySizesAndValidatesDateRange() {
        var f = fixture(false);
        String media = upload(f,"私有统计媒体".getBytes(StandardCharsets.UTF_8));
        create(f,List.of(node("textFile",Map.of("resourceId",media),null)),List.of());
        build(submit(f,"MEDIA","CANVAS"));
        jdbc.update("INSERT INTO afv_canvas_member(canvas_id,user_id,can_edit,can_execute,can_manage,granted_by) VALUES(?,?,FALSE,FALSE,FALSE,?)",f.canvas(),f.id()+1,f.id());
        Instant from = Instant.now().minusSeconds(60), until = Instant.now().plusSeconds(1);
        var owner = ok(activity(f,from,until,f.owner()));
        assertThat(owner.at("/deliveries/bytes").asLong()).isPositive();
        var viewer = ok(activity(f,from,until,f.other()));
        assertThat(viewer.at("/deliveries/total").asInt()).isEqualTo(1);
        assertThat(viewer.at("/deliveries/bytes").asLong()).isZero();
        assertThat(viewer.at("/deliveries/unavailable").asInt()).isEqualTo(1);
        assertThat(activity(f,until,from,f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(activity(f,from.minus(Duration.ofDays(367)),until,f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }

    private ResponseEntity<JsonNode> activity(Fixture f, Instant from, Instant until, HttpHeaders actor) {
        return http.exchange(base(f)+"/activity-statistics?from="+from+"&until="+until,HttpMethod.GET,new HttpEntity<>(actor),JsonNode.class);
    }

    @Test void alreadyImportedLegacyMediaPacksWithoutRefetchingAndOrdinaryUrlsAreNotTreatedAsMedia() throws Exception {
        var f = fixture(true);
        Path root = legacyRoot.path();
        assertThat(root.toAbsolutePath().normalize().startsWith(FILES)).isTrue();
        Files.createDirectories(root);
        String name = UUID.randomUUID()+".png";
        Path image = root.resolve(name);
        ImageIO.write(new BufferedImage(2,2,BufferedImage.TYPE_INT_RGB),"png",image.toFile());
        byte[] bytes = Files.readAllBytes(image);
        String foreignId=UUID.randomUUID().toString();
        jdbc.update("UPDATE afv_project SET cover_url=?,properties=JSON_OBJECT('callbackUrl','https://example.test/workflow-callback','resourceId',?) WHERE id=?", "/media/"+name,foreignId,f.id());
        assertThat(post(f,"/exports/preview",options(f,"ZIP","PROJECT")).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        var before = ok(post(f,"/exports/preview",options(f,"JSON","PROJECT")));
        assertThat(before.path("warnings")).hasSize(1);
        var source = new MediaSourceReference(CanvasValueType.PROJECT,f.id(),0,"coverUrl");
        var imported = sourceImports.importSource(source,ResourceCommandContext.user(new CanvasActor(f.id(),null),null));
        Files.delete(image);
        String job = submit(f,"ZIP","PROJECT");
        build(job);
        var manifest = jobs.get(job,false).manifest();
        assertThat(manifest.warnings()).isEmpty();
        assertThat(manifest.media()).hasSize(1);
        var project = manifest.values().stream().filter(value -> value.type()==CanvasValueType.PROJECT).findFirst().orElseThrow();
        assertThat(project.data().path("coverUrl").asText()).isEqualTo("/media/"+name);
        assertThat(project.data().at("/properties/resourceId").asText()).isEqualTo(foreignId);
        assertThat(project.data().at("/_exportMediaBindings/coverUrl").asText()).isEqualTo("media://"+imported.id());
        assertThat(jdbc.queryForObject("SELECT cover_url FROM afv_project WHERE id=?",String.class,f.id())).isEqualTo("/media/"+name);
        boolean packed = false;
        try (var zip = new ZipInputStream(new ByteArrayInputStream(download(f,job)),StandardCharsets.UTF_8)) {
            for (var entry=zip.getNextEntry();entry!=null;entry=zip.getNextEntry()) {
                if (entry.getName().startsWith("media/"+imported.id())) { assertThat(zip.readAllBytes()).isEqualTo(bytes); packed=true; }
            }
        }
        assertThat(packed).isTrue();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_source WHERE resource_id=?",Integer.class,imported.id())).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_reference WHERE media_id=? AND anchor_kind='CANVAS_EXPORT' AND anchor_id=?",
                Integer.class, imported.id(), job)).isEqualTo(1);
    }

    @Test void historicalDeliverySurvivesDeletedParentsButStillRequiresCurrentProjectPermissionAndCorrectHierarchy() {
        var f = fixture(true);
        jdbc.update("INSERT INTO afv_script_episode(id,script_id,episode_number,title) VALUES(?,?,1,'交付时的第一集')",f.id(),f.id());
        jdbc.update("INSERT INTO afv_script_scene_item(id,script_id,episode_id,scene_heading,scene_description) VALUES(?,?,?,'灯塔','历史场次内容')",f.id(),f.id(),f.id());
        String job = submit(f,"JSON","PROJECT"); build(job);
        byte[] original = download(f,job);
        ok(post(f,"/resource-commands/delete",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","scriptEpisode",
                "resourceId",f.id(),"expectedVersion",0)));
        assertThat(download(f,job)).isEqualTo(original);
        assertThat(new String(original,StandardCharsets.UTF_8)).contains("历史场次内容");
        assertThat(http.exchange(base(f)+"/resource-commands/scriptScene/"+f.id(),HttpMethod.GET,new HttpEntity<>(f.owner()),JsonNode.class).getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,4)",f.id(),f.id()+1);
        var viewer = new Fixture(f.id(),f.canvas(),f.other(),f.owner());
        assertThat(download(viewer,job)).isEqualTo(original);
        jdbc.update("UPDATE afv_project_member SET deleted=TRUE WHERE project_id=? AND user_id=?",f.id(),f.id()+1);
        assertThat(post(viewer,"/exports/"+job+"/access",Map.of()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var other = fixture(true);
        jdbc.update("INSERT INTO afv_script_episode(id,script_id,title) VALUES(?,?,'另一项目分集')",other.id(),other.id());
        jdbc.update("UPDATE afv_script_scene_item SET episode_id=? WHERE id=?",other.id(),f.id());
        assertThat(post(f,"/exports/"+job+"/access",Map.of()).getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }

    @Test void historicalIndependentAssetChildUsesTheActualOwnerEvenWhenItsParentIsDeleted() {
        var f = fixture(false);
        jdbc.update("INSERT INTO afv_asset(id,name,type,user_id,owner_type,owner_id,deleted) VALUES(?,'旧独立资产','character',?,1,?,TRUE)",f.id(),f.id(),f.id());
        jdbc.update("INSERT INTO afv_asset_item(id,asset_id,name) VALUES(?,?,'旧变体')",f.id(),f.id());
        assertThat(resourceQueries.canReadHistorical(CanvasValueType.ASSET_ITEM,f.id(),new CanvasActor(f.id(),null))).isTrue();
        assertThat(resourceQueries.canReadHistorical(CanvasValueType.ASSET_ITEM,f.id(),new CanvasActor(f.id()+1,null))).isFalse();
        assertThatThrownBy(() -> resourceQueries.get(CanvasValueType.ASSET_ITEM,f.id(),new CanvasActor(f.id(),null))).isInstanceOf(BusinessException.class);
    }

    @Test void mediaCopyOwnsItsPermissionAndKeepsTheSharedCiphertextAfterTheOriginalIsDeleted() {
        var f=fixture(false);
        byte[] content="只复制一次的不可变媒体".getBytes(StandardCharsets.UTF_8);
        String original=upload(f,content);
        long project=ok(http.postForEntity("/api/project",new HttpEntity<>(Map.of("requestId",UUID.randomUUID().toString(),
                "fields",Map.of("name","独立媒体副本项目")),f.owner()),JsonNode.class)).path("id").asLong();
        long otherProject=ok(http.postForEntity("/api/project",new HttpEntity<>(Map.of("requestId",UUID.randomUUID().toString(),
                "fields",Map.of("name","没有来源权限的项目")),f.other()),JsonNode.class)).path("id").asLong();
        int version=ok(http.exchange("/api/media-resources/"+original,HttpMethod.GET,new HttpEntity<>(f.owner()),JsonNode.class)).path("version").asInt();
        var writes=new TransactionTemplate(transactionManager);
        assertThatThrownBy(() -> writes.execute(ignored -> mediaCopies.copy(original,version,otherProject,UUID.randomUUID().toString(),new CanvasActor(f.id()+1,null))))
                .isInstanceOfSatisfying(BusinessException.class,failure -> assertThat(failure.getCode()).isEqualTo(403));
        String operation=UUID.randomUUID().toString();
        var copied=writes.execute(ignored -> mediaCopies.copy(original,version,project,operation,new CanvasActor(f.id(),null)));
        assertThat(copied).isNotNull();
        assertThat(copied.id()).isNotEqualTo(original);
        assertThat(copied.projectId()).isEqualTo(project);
        assertThat(jdbc.queryForObject("SELECT blob_id FROM afv_media_resource WHERE id=?",String.class,copied.id()))
                .isEqualTo(jdbc.queryForObject("SELECT blob_id FROM afv_media_resource WHERE id=?",String.class,original));
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_blob WHERE resource_id IN (?,?)",Integer.class,original,copied.id())).isEqualTo(1);
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,4)",project,f.id()+1);
        assertThat(http.postForEntity("/api/media-resources/"+original+"/access",new HttpEntity<>(Map.of(),f.other()),JsonNode.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        jdbc.update("UPDATE afv_media_resource SET state='DELETED',version=version+1 WHERE id=?",original);
        var ticket=ok(http.postForEntity("/api/media-resources/"+copied.id()+"/access",new HttpEntity<>(Map.of(),f.other()),JsonNode.class));
        assertThat(http.getForEntity(ticket.path("url").asText(),byte[].class).getBody()).isEqualTo(content);
        var replay=writes.execute(ignored -> mediaCopies.copy(original,version,project,operation,new CanvasActor(f.id(),null)));
        assertThat(replay.id()).isEqualTo(copied.id());
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_copy WHERE media_id=? AND source_media_id=?",Integer.class,copied.id(),original)).isEqualTo(1);
        jdbc.update("UPDATE afv_project_member SET deleted=TRUE WHERE project_id=? AND user_id=?",project,f.id()+1);
        assertThat(http.getForEntity(ticket.path("url").asText(),byte[].class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }
    @Test void deliveryIncludesFrozenTextAndCopiedSnapshotWithoutDependingOnTheOriginalRun() {
        var f = fixture(false);
        var source = node("text", Map.of("content", "交付保留的旧文案"), null);
        var target = node("textCombine", Map.of(), null);
        var edge = edge(source, "text", target, "texts");
        create(f, List.of(source, target), List.of(edge));
        var run = run(f, "ALL", List.of());
        JsonNode result = null;
        for (var item : run.path("nodes")) if (item.path("nodeId").asText().equals(source.get("id"))) result = item;
        assertThat(result).isNotNull();
        edge.put("output", Map.of("nodeRunId", result.path("id").asText(), "outputId", result.at("/outputs/values/text/0/id").asText()));
        ok(post(f, "/commands", Map.of("requestId", UUID.randomUUID().toString(), "operations", List.of(
                Map.of("type", "canvas.node.patch", "nodeId", source.get("id"), "expectedConfigVersion", 0, "configPatch", Map.of("content", "当前上游已修改")),
                Map.of("type", "canvas.edge.replace", "edge", edge, "expectedVersion", 0)))));
        String checkpoint = UUID.randomUUID().toString(), copied = UUID.randomUUID().toString();
        ok(post(f, "/checkpoints", Map.of("requestId", UUID.randomUUID().toString(), "id", checkpoint, "title", "固定交付", "expectedRevision", revision(f))));
        ok(post(f, "/checkpoints/"+checkpoint+"/fork", Map.of("requestId", UUID.randomUUID().toString(), "canvasId", copied, "title", "固定交付副本")));
        var copy = new Fixture(f.id(), copied, f.owner(), f.other());
        for (Fixture canvas : List.of(f, copy)) {
            String job = submit(canvas, "JSON", "CANVAS"); build(job);
            var manifest = jobs.get(job, false).manifest();
            assertThat(manifest.schemaVersion()).isEqualTo(2);
            assertThat(manifest.fixedOutputs()).hasSize(1);
            assertThat(manifest.fixedOutputs().values().iterator().next().text()).isEqualTo("交付保留的旧文案");
            assertThat(manifest.fixedOutputs().keySet()).containsExactly(manifest.graph().edges().getFirst().id());
            assertThat(writer.markdown(manifest)).contains("固定连线结果", "交付保留的旧文案", "当前上游已修改");
            assertThat(new String(download(canvas, job), StandardCharsets.UTF_8)).contains("fixedOutputs", "交付保留的旧文案");
        }
        var legacy = json.object(jobs.get(submit(copy,"JSON","CANVAS"),false).manifest());
        legacy.put("schemaVersion", 1); legacy.remove("fixedOutputs");
        assertThat(json.hash(json.read(legacy.toString(),CanvasExportData.Manifest.class))).isEqualTo(json.hash(legacy));
    }

    @Test void fixedHistoricalBusinessMediaUsesItsOldImportedMappingAndKeepsCurrentPermissionChecks() throws Exception {
        var f = fixture(true);
        Files.createDirectories(FILES);
        String name = "historical-cover-"+UUID.randomUUID()+".png";
        Path image = legacyRoot.path().resolve(name);
        ImageIO.write(new BufferedImage(8,8,BufferedImage.TYPE_INT_RGB),"png",image.toFile());
        byte[] bytes = Files.readAllBytes(image);
        jdbc.update("UPDATE afv_project SET cover_url=? WHERE id=?", "/media/"+name, f.id());
        var imported = sourceImports.importSource(new MediaSourceReference(CanvasValueType.PROJECT,f.id(),0,"coverUrl"),
                ResourceCommandContext.user(new CanvasActor(f.id(),null),null));
        var source = node("project", Map.of(), Map.of("type","project","id",f.id(),"version",0));
        var target = node("collection", Map.of("valueType","project"), null);
        var edge = edge(source,"project",target,"items");
        create(f,List.of(source,target),List.of(edge));
        var run = run(f,"ALL",List.of());
        JsonNode original = null;
        for (var item : run.path("nodes")) if (item.path("nodeId").asText().equals(source.get("id"))) original=item;
        assertThat(original).isNotNull();
        edge.put("output",Map.of("nodeRunId",original.path("id").asText(),"outputId",original.at("/outputs/values/project/0/id").asText()));
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(
                Map.of("type","canvas.edge.replace","edge",edge,"expectedVersion",0)))));
        ok(post(f,"/resource-commands",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","project","resourceId",f.id(),
                "expectedVersion",0,"patch",Map.of("coverUrl","","description","后来修改的项目说明"))));
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM afv_resource_event WHERE project_id=? AND published=FALSE",Integer.class,f.id())).isZero());
        Files.delete(image);
        String job = submit(f,"ZIP","CANVAS"); build(job);
        var manifest = jobs.get(job,false).manifest();
        assertThat(manifest.fixedOutputs().get(edge.get("id")).resource().version()).isZero();
        assertThat(manifest.fixedOutputs().get(edge.get("id")).data().at("/_exportMediaBindings/coverUrl").asText()).isEqualTo("media://"+imported.id());
        assertThat(manifest.media()).hasSize(1);
        var forkPreview=ok(http.exchange(base(f)+"/project-forks/preview?expectedRevision="+revision(f),HttpMethod.GET,new HttpEntity<>(f.owner()),JsonNode.class));
        String copiedCanvas=UUID.randomUUID().toString();
        var fork=ok(post(f,"/project-forks",Map.of("requestId",UUID.randomUUID().toString(),"sourceCanvasId",f.canvas(),
                "canvasId",copiedCanvas,"title","保留旧封面的独立项目","expectedRevision",forkPreview.path("revision").asLong(),"digest",forkPreview.path("digest").asText()))).path("result");
        var copied=new Fixture(f.id(),copiedCanvas,f.owner(),f.other());
        var copyGraph=ok(http.exchange(base(copied),HttpMethod.GET,new HttpEntity<>(f.owner()),JsonNode.class)).path("graph");
        var fixed=ok(http.exchange(base(copied)+"/fixed-outputs/"+copyGraph.path("edges").get(0).path("id").asText(),HttpMethod.GET,new HttpEntity<>(f.owner()),JsonNode.class));
        String alias=jdbc.queryForObject("SELECT id FROM afv_media_resource WHERE project_id=?",String.class,fork.path("id").asLong());
        assertThat(alias).isNotEqualTo(imported.id());
        assertThat(fixed.path("data").path("coverUrl").asText()).isEqualTo("media://"+alias);
        assertThat(fixed.path("data").has("_exportMediaBindings")).isFalse();
        boolean packed = false;
        try (var zip = new ZipInputStream(new ByteArrayInputStream(download(f,job)),StandardCharsets.UTF_8)) {
            for (var entry=zip.getNextEntry();entry!=null;entry=zip.getNextEntry())
                if (entry.getName().startsWith("media/"+imported.id())) { assertThat(zip.readAllBytes()).isEqualTo(bytes); packed=true; }
        }
        assertThat(packed).isTrue();
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,4)",f.id(),f.id()+1);
        var viewer = new Fixture(f.id(),f.canvas(),f.other(),f.owner());
        assertThat(download(viewer,job)).isNotEmpty();
        jdbc.update("UPDATE afv_project_member SET deleted=TRUE WHERE project_id=? AND user_id=?",f.id(),f.id()+1);
        assertThat(post(viewer,"/exports/"+job+"/access",Map.of()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    private Map<String,Object> edge(Map<String,Object> source, String port, Map<String,Object> target, String input) {
        return new LinkedHashMap<>(Map.of("id",UUID.randomUUID().toString(),"kind","execution","sourceNodeId",source.get("id"),
                "sourcePort",port,"targetNodeId",target.get("id"),"targetPort",input,"inputIndex",0,"version",0));
    }
    private JsonNode run(Fixture f, String mode, List<String> nodes) {
        var scope=Map.of("userId",f.id());
        var plan=ok(post(f,"/run-plans",Map.of("expectedRevision",revision(f),"mode",mode,"nodeIds",nodes,"reuse",Map.of(),"scope",scope,
                "budget",Map.of("maxNodes",10,"maxImages",0,"maxVideoSeconds",0,"maxParallel",2,"maxBatchItems",10))));
        assertThat(plan.path("issues")).isEmpty();
        String id=ok(post(f,"/runs",Map.of("requestId",UUID.randomUUID().toString(),"planId",plan.path("id").asText(),
                "planHash",plan.path("hash").asText(),"scope",scope))).path("id").asText();
        await().atMost(Duration.ofSeconds(15)).untilAsserted(() -> assertThat(ok(http.exchange("/api/canvas-runs/"+id,
                HttpMethod.GET,new HttpEntity<>(f.owner()),JsonNode.class)).path("status").asText()).isEqualTo("SUCCEEDED"));
        return ok(http.exchange("/api/canvas-runs/"+id,HttpMethod.GET,new HttpEntity<>(f.owner()),JsonNode.class));
    }
    private Map<String,Object> node(String kind, Map<String,Object> config, Map<String,Object> resource) {
        var node = new HashMap<String,Object>(); node.put("id", UUID.randomUUID().toString()); node.put("kind",kind); node.put("schemaVersion",1);
        node.put("title","交付测试节点"); node.put("config",config); node.put("resource",resource); node.put("configVersion",0); node.put("layoutVersion",0);
        node.put("layout",Map.of("x",20,"y",30,"width",300,"locked",false,"collapsed",false)); return node;
    }
    private void create(Fixture f, List<Map<String,Object>> nodes, List<Map<String,Object>> edges) {
        var operations = new ArrayList<Map<String,Object>>(); nodes.forEach(node -> operations.add(Map.of("type","canvas.node.create","node",node)));
        edges.forEach(edge -> operations.add(Map.of("type","canvas.edge.create","edge",edge)));
        ok(post(f, "/commands", Map.of("requestId",UUID.randomUUID().toString(),"operations",operations)));
    }
    private long revision(Fixture f) { return ok(http.exchange(base(f), HttpMethod.GET, new HttpEntity<>(f.owner()), JsonNode.class)).path("document").path("revision").asLong(); }
    private Map<String,Object> options(Fixture f, String format, String scope) { return Map.of("format",format,"scope",scope,"nodeIds",List.of(),"expectedRevision",revision(f)); }
    private String submit(Fixture f, String format, String scope) {
        var options = options(f,format,scope); var preview = ok(post(f,"/exports/preview",options));
        return ok(post(f,"/exports",Map.of("requestId",UUID.randomUUID().toString(),"options",options,"previewHash",preview.path("hash").asText()))).path("id").asText();
    }
    private void build(String id) { var claim = transactions.claim(id); assertThat(claim).isNotNull(); worker().execute(claim); assertThat(jobs.get(id,false).status()).as(jobs.get(id,false).error()).isEqualTo("SUCCEEDED"); }
    private CanvasExportWorker worker() { return new CanvasExportWorker(jobs,transactions,writer,blobs); }
    private byte[] download(Fixture f, String id) {
        String url = ok(post(f,"/exports/"+id+"/access",Map.of())).path("url").asText();
        var response = http.getForEntity(url,byte[].class); assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK); return response.getBody();
    }
    private String upload(Fixture f, byte[] bytes) {
        return upload(f, bytes, "私有.txt");
    }
    private String upload(Fixture f, byte[] bytes, String filename) {
        var form = new LinkedMultiValueMap<String,Object>();
        form.add("file", new ByteArrayResource(bytes) {
            @Override
            public String getFilename() {
                return filename;
            }
        });
        form.add("requestId",UUID.randomUUID().toString());
        var headers = new HttpHeaders(); headers.putAll(f.owner()); headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        return ok(http.postForEntity("/api/media-resources/upload",new HttpEntity<>(form,headers),JsonNode.class)).path("id").asText();
    }
    private HttpHeaders headers(long id) { var result = new HttpHeaders(); result.setBearerAuth(tokens.createToken(id,"export-"+id,null).getAccessToken()); result.setContentType(MediaType.APPLICATION_JSON); return result; }
    private String base(Fixture f) { return "/api/canvases/"+f.canvas(); }
    private ResponseEntity<JsonNode> post(Fixture f, String path, Object body) { return http.postForEntity(base(f)+path,new HttpEntity<>(body,f.owner()),JsonNode.class); }
    private JsonNode ok(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull(); assertThat(response.getBody().path("code").asInt()).as(response.toString()).isZero(); return response.getBody().path("data");
    }
}
