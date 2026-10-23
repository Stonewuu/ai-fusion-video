package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelKey;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.awaitility.Awaitility.await;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/** 同一画布从故事到正式业务、首尾帧、视频、配乐音效、混音与成片下载的连续集成验收。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class CanvasFilmJourneyIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "canvas-film-" + UUID.randomUUID()).toAbsolutePath();
    static final long PROJECT = 610000;
    static final long TEXT_MODEL = 610001, IMAGE_MODEL = 610002, VIDEO_MODEL = 610003, AUDIO_API = 610004;
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
    @Autowired ObjectMapper mapper;
    @Autowired CanvasRunService runs;
    @MockitoBean AiProviderService providers;
    private final String canvas = id();
    private final HttpHeaders headers = new HttpHeaders();
    private final List<String> runIds = new ArrayList<>();

    @Test void storyToFilmKeepsFormalContentCandidateChoicesAudioAndDownloadedOrderTraceable() throws Exception {
        try (var provider = new CanvasFilmProvider(mapper, FILES.resolve("provider"))) {
            initialize(provider.url());
            var model = new CanvasFilmScenario(mapper, PROJECT);
            when(providers.agentScopeModelFingerprint(any(AiModel.class))).thenAnswer(call -> AgentKernelKey.modelRecordFingerprint(call.getArgument(0)));
            when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call ->
                    new AiProviderService.AgentScopeModelProvision(model, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
            String story = id(), scriptWriter = id();
            create(List.of(node(story, "story", Map.of("content", "阿青穿蓝色雨衣在灯塔捡到一封信。画风：温暖悬疑、手绘电影质感。")),
                    node(scriptWriter, "storyToScript", Map.of("agentType", "story_to_script", "modelId", TEXT_MODEL,
                            "instruction", "创作并保存一集一场剧本及对白", "maxAgentRuns", 2))),
                    List.of(edge(story, "text", scriptWriter, "story", 0)));
            String storyRun = start(List.of(scriptWriter), Map.of(), 0, 0);
            // 专业节点和子任务执行已提交的计划，继续验证实际写回，不再等待逐工具确认。
            var scriptDone = awaitState(storyRun, "SUCCEEDED");
            assertThat(result(scriptDone, scriptWriter).at("/outputs/values/script/0/resource/id").asLong()).isEqualTo(PROJECT);
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_script_scene_item WHERE script_id=? AND deleted=FALSE", Integer.class, PROJECT)).isEqualTo(1);

            model.phase(CanvasFilmScenario.Phase.ASSETS);
            String extract = id();
            create(List.of(node(extract, "assetExtract", Map.of("agentType", "canvas_asset_extractor", "modelId", TEXT_MODEL,
                    "instruction", "提取角色与场景并建立子资产", "maxAgentRuns", 1))), List.of(edge(scriptWriter, "script", extract, "script", 0)));
            String assetRun = start(List.of(extract), Map.of(scriptWriter, result(scriptDone, scriptWriter).path("id").asText()), 0, 0);
            var assetDone = awaitState(assetRun, "SUCCEEDED");
            assertThat(result(assetDone, extract).at("/outputs/values/assets")).hasSize(2);
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset_item i JOIN afv_asset a ON a.id=i.asset_id WHERE a.project_id=? AND i.deleted=FALSE", Integer.class, PROJECT)).isEqualTo(2);

            model.phase(CanvasFilmScenario.Phase.SHOTS);
            String episode = id(), storyboardWriter = id();
            create(List.of(resource(episode, "scriptEpisode", model.episode()), node(storyboardWriter, "agent", Map.of(
                    "agentType", "episode_storyboard_writer", "inputType", "scriptEpisode", "resultType", "shot", "modelId", TEXT_MODEL,
                    "instruction", "为此分集保存两个镜头，保持角色与场景子资产关联", "maxAgentRuns", 1))),
                    List.of(edge(episode, "episode", storyboardWriter, "inputs", 0)));
            String boardRun = start(List.of(storyboardWriter), Map.of(), 0, 0);
            assertThat(result(awaitState(boardRun, "SUCCEEDED"), storyboardWriter).at("/outputs/values/results")).hasSize(2);

            patch("scriptScene", model.scene(), currentVersion("afv_script_scene_item", model.scene()), Map.of("dialogues",
                    List.of(Map.of("type", 1, "character_name", "阿青", "content", "我会沿着灯光找到归途。", "sortOrder", 0))));
            var shotNodes = List.of(id(), id());
            for (int index = 0; index < 2; index++) {
                long shot = model.shots().get(index);
                patch("shot", shot, 0, Map.of("cameraMovement", index == 0 ? "缓慢推近" : "横移"));
            }
            create(List.of(resource(shotNodes.get(0), "shot", model.shots().get(0)), resource(shotNodes.get(1), "shot", model.shots().get(1))), List.of());
            generateAndChooseFrames(shotNodes, "firstFrameImageUrl");
            generateAndChooseFrames(shotNodes, "lastFrameImageUrl");
            for (long shot : model.shots()) {
                var row = jdbc.queryForMap("SELECT first_frame_image_url,last_frame_image_url,version FROM afv_storyboard_item WHERE id=?", shot);
                assertThat(row.get("first_frame_image_url").toString()).startsWith("media://");
                assertThat(row.get("last_frame_image_url").toString()).startsWith("media://");
                assertThat(row.get("version")).isEqualTo(3);
            }

            var generate = List.of(id(), id());
            var videoNodes = new ArrayList<Map<String, Object>>();
            var videoEdges = new ArrayList<Map<String, Object>>();
            for (int index = 0; index < 2; index++) {
                videoNodes.add(node(generate.get(index), "videoGeneration", Map.of("modelId", VIDEO_MODEL, "duration", 4,
                        "count", 1, "ratio", "16:9", "resolution", "1280x720")));
                videoEdges.add(edge(shotNodes.get(index), "prompt", generate.get(index), "prompt", 0));
                videoEdges.add(edge(shotNodes.get(index), "firstFrame", generate.get(index), "firstFrame", 0));
            }
            create(videoNodes, videoEdges);
            String videoRun = start(generate, Map.of(), 0, 8);
            var videoDone = awaitState(videoRun, "SUCCEEDED");
            var videoMedia = new ArrayList<String>();
            var reuse = new HashMap<String, String>();
            for (int index = 0; index < 2; index++) {
                var generated = result(videoDone, generate.get(index));
                String media = generated.at("/outputs/values/videos/0/mediaId").asText(); videoMedia.add(media);
                reuse.put(generate.get(index), generated.path("id").asText());
                patch("shot", model.shots().get(index), 3, Map.of("videoUrl", "media://" + media));
            }
            String compose = id(), music = id(), sfx = id(), mix = id(), delivery = id();
            create(List.of(node(compose, "videoCompose", Map.of("resolution", "160x90", "frameRate", 24)),
                    node(music, "musicGeneration", Map.of("apiConfigId", AUDIO_API, "operation", "music", "prompt", "温暖悬疑配乐")),
                    node(sfx, "sfxGeneration", Map.of("apiConfigId", AUDIO_API, "operation", "sfx", "prompt", "雨声和脚步声")),
                    node(mix, "audioMix", Map.of("musicVolume", 0.3, "sfxVolume", 0.2, "originalVolume", 0.5)),
                    node(delivery, "export", Map.of("valueType", "video", "format", "media"))), List.of(
                    edge(generate.get(1), "videos", compose, "videos", 0), edge(generate.get(0), "videos", compose, "videos", 1),
                    edge(compose, "video", music, "video", 0), edge(compose, "video", sfx, "video", 0),
                    edge(compose, "video", mix, "video", 0), edge(music, "audio", mix, "music", 0), edge(sfx, "audio", mix, "sfx", 0),
                    edge(mix, "video", delivery, "items", 0)));
            String filmRun = start(List.of(delivery), reuse, 0, 0);
            var done = awaitState(filmRun, "SUCCEEDED");
            var ordered = result(done, compose).at("/inputs/values/videos");
            assertThat(ordered).hasSize(2);
            assertThat(ordered.get(0).path("mediaId").asText()).isEqualTo(videoMedia.get(1));
            assertThat(ordered.get(1).path("mediaId").asText()).isEqualTo(videoMedia.get(0));
            var mixed = result(done, mix).at("/outputs/values/video/0/data/content");
            assertThat(mixed.path("hasAudio").asBoolean()).isTrue();
            assertThat(mixed.path("durationMs").asLong()).isBetween(7900L, 8300L);
            assertThat(result(done, music).path("status").asText()).isEqualTo("SUCCEEDED");
            assertThat(result(done, sfx).path("status").asText()).isEqualTo("SUCCEEDED");
            String export = result(done, delivery).path("externalId").asText();
            var access = post(base() + "/exports/" + export + "/access", Map.of());
            var download = http.getForEntity(access.path("url").asText(), byte[].class);
            assertThat(download.getStatusCode()).isEqualTo(HttpStatus.OK);
            assertThat(download.getBody()).isNotEmpty();
            Path film = FILES.resolve("delivered-film.mp4"); Files.write(film, download.getBody());
            Color first = frameColor(film, "0.2"), second = frameColor(film, "4.2");
            assertThat(first.getBlue()).isGreaterThan(180); assertThat(first.getRed()).isLessThan(50);
            assertThat(second.getRed()).isGreaterThan(180); assertThat(second.getBlue()).isLessThan(50);
            assertThat(provider.imagePosts).hasValue(4); assertThat(provider.videoPosts).hasValue(2); assertThat(provider.audioPosts).hasValue(2);
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_run WHERE canvas_id=? AND status='SUCCEEDED'", Integer.class, canvas)).isEqualTo(7);
            assertThat(jdbc.queryForObject("SELECT dialogues FROM afv_script_scene_item WHERE id=?", String.class, model.scene())).contains("我会沿着灯光找到归途");
            assertThat(jdbc.queryForObject("SELECT camera_movement FROM afv_storyboard_item WHERE id=?", String.class, model.shots().getFirst())).isEqualTo("缓慢推近");
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE canvas_id=? AND origin='agent'", Integer.class, canvas)).isGreaterThanOrEqualTo(5);
            Files.writeString(Path.of("target/g-story-to-film-journey-results.json"), mapper.writerWithDefaultPrettyPrinter().writeValueAsString(Map.ofEntries(
                    Map.entry("environment", "isolated MySQL/Redis, actual AgentScope SDK and FFmpeg, local HTTP media providers"),
                    Map.entry("scriptEpisodes", 1), Map.entry("scriptScenes", 1), Map.entry("assetsAndVariants", 2), Map.entry("shots", 2),
                    Map.entry("runs", runIds), Map.entry("modelRequests", model.calls()), Map.entry("imageCandidates", 8),
                    Map.entry("imagePosts", provider.imagePosts.get()), Map.entry("videoPosts", provider.videoPosts.get()),
                    Map.entry("audioPosts", provider.audioPosts.get()), Map.entry("downloadBytes", download.getBody().length),
                    Map.entry("durationMs", mixed.path("durationMs").asLong()), Map.entry("hasAudio", true),
                    Map.entry("orderedBlueThenRedFramesVerified", true), Map.entry("humanDialogueAndCameraEditsPreserved", true))));
        }
    }

    private void initialize(String url) {
        jdbc.update("INSERT INTO sys_user(id,username,password) VALUES(?,'film-journey','test')", PROJECT);
        jdbc.update("INSERT INTO afv_project(id,name,owner_type,owner_id,art_style_description) VALUES(?,'故事到成片验收',1,?,'温暖悬疑、手绘电影质感')", PROJECT, PROJECT);
        jdbc.update("INSERT INTO afv_script(id,project_id,title) VALUES(?,?,'雨夜灯塔')", PROJECT, PROJECT);
        jdbc.update("INSERT INTO afv_storyboard(id,project_id,script_id,title) VALUES(?,?,?,'雨夜灯塔分镜')", PROJECT, PROJECT, PROJECT);
        jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('成片验收存储','local',?,TRUE)", FILES.toString());
        headers.setBearerAuth(tokens.createToken(PROJECT, "film-journey", null).getAccessToken()); headers.setContentType(MediaType.APPLICATION_JSON);
        post("/api/canvases", Map.of("id", canvas, "title", "故事到成片", "projectId", PROJECT));
        for (long api : List.of(TEXT_MODEL, IMAGE_MODEL, VIDEO_MODEL, AUDIO_API)) jdbc.update("""
                INSERT INTO afv_api_config(id,name,platform,text_protocol,image_protocol,video_protocol,api_url,api_key)
                VALUES(?,'成片验收本机接口',?,'openai_compatible','openai','openai',?,'test-key')
                """, api, api == AUDIO_API ? "sonilo" : "openai_compatible", url);
        insertModel(TEXT_MODEL, "film-script", 1, "openai_compatible", "{}");
        insertModel(IMAGE_MODEL, "gpt-image-2", 2, "openai", "{\"defaultWidth\":1280,\"defaultHeight\":720}");
        insertModel(VIDEO_MODEL, "sora-2", 3, "openai", "{\"supportFirstFrame\":true,\"referenceImageInputFormats\":[\"data_uri\"],\"maxImageInputs\":1,\"defaultDuration\":4}");
    }
    private void insertModel(long id, String code, int type, String protocol, String config) {
        jdbc.update("""
                INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels)
                VALUES(?,'成片验收模型',?,?,?,?,?,2,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())
                """, id, code, type, protocol, id, config);
    }
    private void generateAndChooseFrames(List<String> shots, String field) {
        var creates = new ArrayList<Map<String, Object>>(); var edges = new ArrayList<Map<String, Object>>();
        var choices = List.of(id(), id()); var writes = List.of(id(), id());
        for (int index = 0; index < 2; index++) {
            String generation = id();
            creates.add(node(generation, "imageGeneration", Map.of("modelId", IMAGE_MODEL, "prompt", "蓝色雨衣、雨夜灯塔 " + field + " " + index,
                    "width", 1280, "height", 720, "count", 2)));
            creates.add(node(choices.get(index), "selection", Map.of("valueType", "image")));
            creates.add(node(writes.get(index), "writeback", Map.of("binding", "shot:" + field, "targetType", "shot", "valueType", "image")));
            edges.add(edge(generation, "images", choices.get(index), "candidates", 0));
            edges.add(edge(choices.get(index), "selected", writes.get(index), "value", 0));
            edges.add(edge(shots.get(index), "shot", writes.get(index), "target", 0));
        }
        create(creates, edges);
        String run = start(writes, Map.of(), 4, 0);
        for (int index = 0; index < 2; index++) {
            final String selected = choices.get(index);
            await().atMost(Duration.ofSeconds(45)).untilAsserted(() -> assertThat(result(view(run), selected).path("status").asText()).isEqualTo("WAITING_SELECTION"));
            var choice = result(view(run), selected);
            assertThat(choice.at("/inputs/values/candidates")).hasSize(2);
            post("/api/canvas-runs/" + run + "/nodes/" + choice.path("id").asText() + "/select", Map.of("expectedRevision", choice.path("revision").asLong(),
                    "outputId", choice.at("/inputs/values/candidates/" + index + "/id").asText(), "scope", scope()));
        }
        for (String write : writes) {
            await().atMost(Duration.ofSeconds(30)).untilAsserted(() -> assertThat(result(view(run), write).path("status").asText()).isEqualTo("WAITING_CONFIRMATION"));
            var pending = result(view(run), write);
            post("/api/canvas-runs/" + run + "/nodes/" + pending.path("id").asText() + "/writeback", Map.of(
                    "expectedRevision", pending.path("revision").asLong(), "proposalHash", pending.at("/writeback/hash").asText(),
                    "action", "APPROVE", "note", "已比较候选并核对目标镜头", "scope", scope()));
        }
        awaitState(run, "SUCCEEDED");
    }
    private Color frameColor(Path film, String seconds) throws Exception {
        Path frame = FILES.resolve("delivered-" + seconds + ".png");
        CanvasFilmProvider.execute(FILES.resolve("delivered-" + seconds + ".log"), List.of("ffmpeg", "-v", "error", "-y", "-ss", seconds,
                "-i", film.toString(), "-frames:v", "1", frame.toString()));
        return new Color(ImageIO.read(frame.toFile()).getRGB(40, 40));
    }
    private int currentVersion(String table, long id) { return jdbc.queryForObject("SELECT version FROM " + table + " WHERE id=?", Integer.class, id); }
    private void patch(String type, long id, int version, Map<String, Object> fields) {
        post(base() + "/resource-commands", Map.of("requestId", id(), "resourceType", type, "resourceId", id, "expectedVersion", version, "patch", fields));
    }
    private Map<String, Object> node(String id, String kind, Map<String, Object> config) {
        return Map.of("id", id, "kind", kind, "title", kind, "schemaVersion", 1, "config", config, "configVersion", 0, "layoutVersion", 0,
                "layout", Map.of("x", 40, "y", 80, "width", 300, "locked", false, "collapsed", false));
    }
    private Map<String, Object> resource(String id, String type, long resourceId) {
        int version = get(base() + "/resource-commands/" + type + "/" + resourceId).at("/header/version").asInt();
        var node = new HashMap<>(node(id, type, Map.of())); node.put("resource", Map.of("type", type, "id", resourceId, "version", version)); return node;
    }
    private Map<String, Object> edge(String source, String sourcePort, String target, String targetPort, int index) {
        return Map.of("id", id(), "kind", "execution", "sourceNodeId", source, "sourcePort", sourcePort,
                "targetNodeId", target, "targetPort", targetPort, "inputIndex", index, "version", 0);
    }
    private void create(List<Map<String, Object>> nodes, List<Map<String, Object>> edges) {
        var commands = new ArrayList<Object>(); nodes.forEach(node -> commands.add(Map.of("type", "canvas.node.create", "node", node)));
        edges.forEach(edge -> commands.add(Map.of("type", "canvas.edge.create", "edge", edge)));
        post(base() + "/commands", Map.of("requestId", id(), "operations", commands));
    }
    private String start(List<String> targets, Map<String, String> reuse, int images, int videoSeconds) {
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM afv_resource_event WHERE project_id=? AND published=FALSE", Integer.class, PROJECT)).isZero());
        var plan = post(base() + "/run-plans", Map.of("expectedRevision", get(base()).at("/document/revision").asLong(),
                "mode", "DEPENDENCIES", "nodeIds", targets, "reuse", reuse, "scope", scope(), "budget", Map.of(
                        "maxNodes", 100, "maxImages", images, "maxVideoSeconds", videoSeconds, "maxParallel", 2, "maxBatchItems", 20)));
        assertThat(plan.path("issues")).as(plan.toString()).isEmpty();
        String run = post(base() + "/runs", Map.of("requestId", id(), "planId", plan.path("id").asText(), "planHash", plan.path("hash").asText(), "scope", scope())).path("id").asText();
        runIds.add(run); return run;
    }
    private JsonNode awaitState(String run, String state) {
        await().atMost(Duration.ofSeconds(90)).pollInterval(Duration.ofMillis(250)).untilAsserted(() -> assertThat(view(run).path("status").asText()).as(view(run).toString()).isEqualTo(state));
        return view(run);
    }
    private JsonNode result(JsonNode run, String node) {
        for (var item : run.path("nodes")) if (item.path("nodeId").asText().equals(node)) return item;
        throw new AssertionError("缺少节点 " + node);
    }
    private JsonNode view(String run) { return get("/api/canvas-runs/" + run); }
    private JsonNode post(String url, Object body) { return ok(http.postForEntity(url, new HttpEntity<>(body, headers), JsonNode.class)); }
    private JsonNode get(String url) { return ok(http.exchange(url, HttpMethod.GET, new HttpEntity<>(headers), JsonNode.class)); }
    private JsonNode ok(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody().path("code").asInt()).as(response.toString()).isZero(); return response.getBody().path("data");
    }
    private String base() { return "/api/canvases/" + canvas; }
    private CanvasActor actor() { return new CanvasActor(PROJECT, null); }
    private Map<String, Object> scope() { return Map.of("userId", PROJECT); }
    private static String id() { return UUID.randomUUID().toString(); }
}
