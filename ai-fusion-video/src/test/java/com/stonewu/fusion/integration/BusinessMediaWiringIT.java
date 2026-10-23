package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.config.ArtStylePresets;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.job.MediaJobQueries;
import com.stonewu.fusion.service.storyboard.VideoComposeService;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.TransactionDefinition;
import org.springframework.transaction.support.TransactionTemplate;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.UUID;
import java.util.Set;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.concurrent.TimeUnit;
import static org.assertj.core.api.Assertions.*;

/** 全应用与真实 HTTP 路由，验证组件装配、私有上传、统一业务命令、事件和媒体访问。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class BusinessMediaWiringIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "business-media-wiring-" + UUID.randomUUID()).toAbsolutePath();
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
    @Autowired MediaIntakeService intake;
    @Autowired VideoComposeService compose;
    @Autowired MediaJobQueries mediaJobs;
    @Autowired PlatformTransactionManager transactions;

    @Test void authenticatedUploadBindingAndRevocationUseRealApplicationWiring() throws Exception {
        Files.createDirectories(FILES);
        jdbc.update("INSERT INTO sys_user(id, username, password) VALUES (771, 'wiring-owner', 'test'), (772, 'wiring-other', 'test')");
        jdbc.update("INSERT INTO sys_user_role(user_id, role_id) SELECT 771, id FROM sys_role WHERE code = 'admin' AND deleted = FALSE");
        jdbc.update("INSERT INTO afv_project(id, name, owner_type, owner_id) VALUES (771, '写回联调', 1, 771)");
        jdbc.update("INSERT INTO afv_storage_config(id, name, type, base_path, is_default) VALUES (771, '受控文件', 'local', ?, TRUE)", FILES.toString());
        var headers = new HttpHeaders();
        headers.setBearerAuth(tokens.createToken(771L, "wiring-owner", null).getAccessToken());
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        var bytes = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(20, 12, BufferedImage.TYPE_INT_RGB), "png", bytes);
        var partHeaders = new HttpHeaders();
        partHeaders.setContentType(MediaType.IMAGE_PNG);
        var body = new LinkedMultiValueMap<String, Object>();
        body.add("file", new HttpEntity<>(new ByteArrayResource(bytes.toByteArray()) {
            @Override public String getFilename() { return "preview.png"; }
        }, partHeaders));
        body.add("requestId", UUID.randomUUID().toString());
        var upload = http.postForEntity("/api/storage/upload", new HttpEntity<>(body, headers), JsonNode.class);
        assertThat(upload.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(upload.getBody().path("code").asInt()).as(upload.toString()).isZero();
        String media = MediaReference.id(upload.getBody().path("data").asText());
        headers.setContentType(MediaType.APPLICATION_JSON);
        var command = Map.of("requestId", UUID.randomUUID().toString(), "resourceType", "project", "resourceId", 771,
                "expectedVersion", 0, "bindings", Map.of("coverUrl", media));
        var bind = http.postForEntity("/api/media-bindings", new HttpEntity<>(command, headers), JsonNode.class);
        assertThat(bind.getStatusCode()).as(bind.toString()).isEqualTo(HttpStatus.OK);
        assertThat(bind.getBody().path("data").path("result").path("version").asInt()).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT cover_url FROM afv_project WHERE id = 771", String.class)).isEqualTo(MediaReference.of(media));
        var replay = http.postForEntity("/api/media-bindings", new HttpEntity<>(command, headers), JsonNode.class);
        assertThat(replay.getBody()).isEqualTo(bind.getBody());
        var signed = http.postForEntity("/api/media-resources/" + media + "/access", new HttpEntity<>(headers), JsonNode.class);
        String url = signed.getBody().path("data").path("url").asText();
        var content = http.getForEntity(url, byte[].class);
        assertThat(content.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(content.getBody()).isEqualTo(bytes.toByteArray());
        var foreign = new HttpHeaders();
        foreign.setBearerAuth(tokens.createToken(772L, "wiring-other", null).getAccessToken());
        assertThat(http.exchange("/api/media-resources/" + media, HttpMethod.GET, new HttpEntity<>(foreign), String.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        String key = ArtStylePresets.getAll().getFirst().getKey();
        var published = http.postForEntity("/api/media-presets/" + key + "/publish",
                new HttpEntity<>(Map.of("requestId", UUID.randomUUID().toString()), headers), JsonNode.class);
        assertThat(published.getStatusCode()).as(published.toString()).isEqualTo(HttpStatus.OK);
        String shared = MediaReference.id(published.getBody().path("data").asText());
        var sharedTicket = http.postForEntity("/api/media-resources/" + shared + "/access", new HttpEntity<>(foreign), JsonNode.class);
        assertThat(sharedTicket.getStatusCode()).isEqualTo(HttpStatus.OK);
        String sharedUrl = sharedTicket.getBody().path("data").path("url").asText();
        assertThat(http.getForEntity(sharedUrl, byte[].class).getStatusCode()).isEqualTo(HttpStatus.OK);
        var forged = http.exchange("/api/system/config", HttpMethod.PUT,
                new HttpEntity<>(Map.of("art_preset_url:" + key, MediaReference.of(media)), headers), JsonNode.class);
        assertThat(forged.getBody().path("code").asInt()).isNotZero();
        jdbc.update("DELETE FROM afv_media_preset_binding WHERE preset_key = ?", key);
        assertThat(http.getForEntity(sharedUrl, byte[].class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var created = http.postForEntity("/api/project", new HttpEntity<>(Map.of("requestId", UUID.randomUUID().toString(),
                "fields", Map.of("name", "业务时区验证")), headers), JsonNode.class);
        assertThat(created.getBody().path("code").asInt()).as(created.toString()).isZero();
        var createdTime = LocalDateTime.parse(created.getBody().path("data").path("createTime").asText());
        assertThat(Duration.between(createdTime, LocalDateTime.now(ZoneId.of("Asia/Shanghai"))).abs()).isLessThan(Duration.ofSeconds(10));
        // 使用真实调度器验证 +08:00 会话中的默认入队时刻，不能只手动调用 worker 掩盖时间错位。
        Path video = FILES.resolve("automatic.mp4");
        Process encoder = new ProcessBuilder("ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                "color=c=blue:s=64x48:r=10:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", video.toString())
                .redirectOutput(ProcessBuilder.Redirect.DISCARD).redirectError(ProcessBuilder.Redirect.DISCARD).start();
        boolean encoded = encoder.waitFor(30, TimeUnit.SECONDS);
        if (!encoded) encoder.destroyForcibly();
        assertThat(encoded).isTrue();
        assertThat(encoder.exitValue()).isZero();
        String input = intake.result(video, "自动执行.mp4", "video/mp4", UUID.randomUUID().toString(), 771L,
                ResourceCommandContext.user(new CanvasActor(771L, null), null)).id();
        jdbc.update("INSERT INTO afv_storyboard(id, project_id) VALUES (771, 771)");
        jdbc.update("INSERT INTO afv_storyboard_episode(id, storyboard_id, episode_number, title) VALUES (771, 771, 1, '自动领取')");
        jdbc.update("INSERT INTO afv_storyboard_scene(id, storyboard_id, episode_id) VALUES (771, 771, 771)");
        jdbc.update("INSERT INTO afv_storyboard_item(id, storyboard_id, storyboard_episode_id, storyboard_scene_id, video_url) VALUES (771, 771, 771, 771, ?)", MediaReference.of(input));
        var submitted = http.postForEntity("/api/storyboard/episode/771/compose-video", new HttpEntity<>(Map.of("version", 0,
                "requestId", UUID.randomUUID().toString(), "includeBgm", false), headers), JsonNode.class);
        assertThat(submitted.getBody().path("code").asInt()).as(submitted.toString()).isZero();
        String jobId = submitted.getBody().path("data").asText();
        JsonNode finished = null;
        long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(45);
        while (System.nanoTime() < until) {
            finished = http.exchange("/api/media-jobs/" + jobId, HttpMethod.GET, new HttpEntity<>(headers), JsonNode.class).getBody().path("data");
            if (!Set.of("QUEUED", "RUNNING", "WAITING").contains(finished.path("status").asText())) break;
            Thread.sleep(250);
        }
        assertThat(finished.path("status").asText()).as(finished.toString()).isEqualTo("SUCCEEDED");
        assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_conversation WHERE conversation_id = ?", String.class, jobId)).isEqualTo("completed");
        assertThat(jdbc.queryForObject("SELECT composed_video_url FROM afv_storyboard_episode WHERE id = 771", String.class)).startsWith("media://");
        // 同一事务提交任务及取消请求，确保真实调度器领取时已可见取消，避免依赖执行速度的竞态测试。
        var cancellation = new TransactionTemplate(transactions);
        cancellation.setIsolationLevel(TransactionDefinition.ISOLATION_READ_COMMITTED);
        String cancelledId = cancellation.execute(transaction -> {
            int version = jdbc.queryForObject("SELECT version FROM afv_storyboard_episode WHERE id = 771", Integer.class);
            var actor = new CanvasActor(771L, null);
            String id = compose.submitCompose(771, version, UUID.randomUUID().toString(), ResourceCommandContext.user(actor, null), false);
            assertThat(mediaJobs.cancel(id, actor)).isTrue();
            return id;
        });
        until = System.nanoTime() + TimeUnit.SECONDS.toNanos(15);
        while (System.nanoTime() < until) {
            finished = http.exchange("/api/media-jobs/" + cancelledId, HttpMethod.GET, new HttpEntity<>(headers), JsonNode.class).getBody().path("data");
            if (finished.path("status").asText().equals("CANCELLED")) break;
            Thread.sleep(250);
        }
        assertThat(finished.path("status").asText()).as(finished.toString()).isEqualTo("CANCELLED");
        assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_conversation WHERE conversation_id = ?", String.class, cancelledId)).isEqualTo("cancelled");
        var cancelledStatus = http.exchange("/api/task-stream/status?taskId=" + cancelledId, HttpMethod.GET, new HttpEntity<>(headers), JsonNode.class);
        assertThat(cancelledStatus.getBody().path("data").asText()).isEqualTo("CANCELLED");
        var cancelledEvents = http.exchange("/api/task-stream/reconnect?taskId=" + cancelledId, HttpMethod.GET, new HttpEntity<>(headers), String.class);
        assertThat(cancelledEvents.getBody()).contains("\"outputType\":\"CANCELLED\"", "\"finished\":true").doesNotContain("\"outputType\":\"ERROR\"");
        jdbc.update("UPDATE sys_user SET status = 0 WHERE id = 771");
        assertThat(http.getForEntity(url, byte[].class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }
}
