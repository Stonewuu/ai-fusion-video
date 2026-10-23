package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.resource.ResourceEventProjectionService;
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
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;
import static org.assertj.core.api.Assertions.assertThat;

/** 使用真实应用装配及 HTTP，覆盖跨业务/画布事务和结构读取权限，不调用生成供应商。 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class CanvasWorkspaceWiringIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "canvas-workspace-wiring-" + UUID.randomUUID()).toAbsolutePath();
    static final AtomicLong IDS = new AtomicLong(9000);
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
    @Autowired ResourceEventProjectionService projection;
    record Fixture(long id, String canvas, HttpHeaders owner, HttpHeaders viewer) { }

    @Test void canvasSearchFindsOlderPagesAndEscapesWildcardsWithoutExposingOtherProjects() {
        var owner = fixture();
        var privateProject = fixture();
        jdbc.update("UPDATE afv_project SET name = '雨夜_100%私密项目' WHERE id = ?", privateProject.id());
        for (int index = 0; index < 31; index++) {
            jdbc.update("INSERT INTO afv_canvas(id,title,project_id,owner_type,owner_id,created_by,update_time) VALUES (?,?,?,1,?,?,UTC_TIMESTAMP(6))",
                    UUID.randomUUID().toString(), "近期草稿 " + index, owner.id(), owner.id(), owner.id());
        }
        String literal = UUID.randomUUID().toString();
        String wildcard = UUID.randomUUID().toString();
        jdbc.update("INSERT INTO afv_canvas(id,title,project_id,owner_type,owner_id,created_by,update_time) VALUES (?,?,?,1,?,?, '2000-01-01')",
                literal, "雨夜_100%完成", owner.id(), owner.id(), owner.id());
        jdbc.update("INSERT INTO afv_canvas(id,title,project_id,owner_type,owner_id,created_by,update_time) VALUES (?,?,?,1,?,?, '2000-01-02')",
                wildcard, "雨夜X100Y完成", owner.id(), owner.id(), owner.id());
        var firstPage = ok(http.exchange("/api/canvases?pageSize=30", HttpMethod.GET, new HttpEntity<>(owner.owner()), JsonNode.class));
        assertThat(firstPage.path("list").toString()).doesNotContain(literal);
        var found = ok(http.exchange("/api/canvases?search={search}&pageSize=30", HttpMethod.GET,
                new HttpEntity<>(owner.owner()), JsonNode.class, "雨夜_100%"));
        assertThat(found.path("total").asInt()).isEqualTo(1);
        assertThat(found.path("list").get(0).path("id").asText()).isEqualTo(literal);
        assertThat(found.path("list").get(0).path("projectName").asText()).isEqualTo("画布测试项目");
        assertThat(found.path("list").get(0).has("document")).isFalse();
        assertThat(found.toString()).doesNotContain(wildcard, "私密项目");
        var denied = http.exchange("/api/canvases?projectId=" + privateProject.id() + "&search=雨夜", HttpMethod.GET,
                new HttpEntity<>(owner.owner()), JsonNode.class);
        assertThat(denied.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void canvasListShowsCurrentProjectNameAndKeepsPersonalAndArchivedFiltersSeparate() {
        var f = fixture();
        jdbc.update("UPDATE afv_project SET name = '灯塔剧本项目' WHERE id = ?", f.id());
        var byProject = ok(http.exchange("/api/canvases?search={search}", HttpMethod.GET, new HttpEntity<>(f.owner()), JsonNode.class, "灯塔剧本项目"));
        assertThat(byProject.path("total").asInt()).isEqualTo(1);
        assertThat(byProject.path("list").get(0).path("projectName").asText()).isEqualTo("灯塔剧本项目");
        String personal = UUID.randomUUID().toString();
        ok(http.postForEntity("/api/canvases", new HttpEntity<>(Map.of("id",personal,"title","独立灵感"), f.owner()), JsonNode.class));
        var personalList = ok(http.exchange("/api/canvases?search={search}", HttpMethod.GET, new HttpEntity<>(f.owner()), JsonNode.class, "独立灵感"));
        assertThat(personalList.path("list").get(0).path("projectName").isNull()).isTrue();
        assertThat(personalList.path("list").get(0).path("projectId").isNull()).isTrue();
        jdbc.update("UPDATE afv_canvas SET archived = TRUE WHERE id = ?", personal);
        assertThat(ok(http.exchange("/api/canvases?search={search}", HttpMethod.GET, new HttpEntity<>(f.owner()), JsonNode.class, "独立灵感")).path("total").asInt()).isZero();
        assertThat(ok(http.exchange("/api/canvases?archived=true&search={search}", HttpMethod.GET, new HttpEntity<>(f.owner()), JsonNode.class, "独立灵感")).path("total").asInt()).isEqualTo(1);
    }

    @Test void mediaNodesCheckTheRealTypeAndProjectAndDoNotBlockUnrelatedLayout() {
        var f = fixture();
        jdbc.update("INSERT INTO afv_storage_config(id, name, type, base_path, is_default) VALUES (?, '测试存储', 'local', ?, TRUE)",f.id(),FILES.toString());
        var multipart = new LinkedMultiValueMap<String,Object>();
        var part = new HttpHeaders();
        part.setContentType(MediaType.TEXT_PLAIN);
        multipart.add("file",new HttpEntity<>(new ByteArrayResource("画布文本素材".getBytes(StandardCharsets.UTF_8)) {
            @Override public String getFilename() {return "canvas.txt";}
        },part));
        multipart.add("requestId",UUID.randomUUID().toString());
        multipart.add("projectId",String.valueOf(f.id()));
        multipart.add("canvasId",f.canvas());
        var headers = new HttpHeaders();
        headers.addAll(f.owner());
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        String media = ok(http.postForEntity("/api/media-resources/upload",new HttpEntity<>(multipart,headers),JsonNode.class)).path("id").asText();
        String node = UUID.randomUUID().toString();
        var layout = Map.of("x",40,"y",80,"width",300,"locked",false,"collapsed",false);
        var invalid = Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.node.create","node",Map.of(
                "id",node,"kind","image","title","错类型","schemaVersion",1,"config",Map.of("resourceId",media),"configVersion",0,"layoutVersion",0,"layout",layout))));
        assertThat(post(f,"/commands",invalid).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(ok(get(f,"",f.owner())).path("graph").path("nodes")).isEmpty();
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.node.create","node",Map.of(
                "id",node,"kind","textFile","title","文本文件","schemaVersion",1,"config",Map.of("resourceId",media),"configVersion",0,"layoutVersion",0,"layout",layout))))));
        var another = fixture();
        jdbc.update("UPDATE afv_project SET owner_id = ? WHERE id = ?", f.id(), another.id());
        var cross = http.postForEntity("/api/canvases/"+another.canvas()+"/commands",new HttpEntity<>(Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.node.create","node",Map.of(
                "id",UUID.randomUUID().toString(),"kind","textFile","title","跨项目引用","schemaVersion",1,"config",Map.of("resourceId",media),"configVersion",0,"layoutVersion",0,"layout",layout)))),f.owner()),JsonNode.class);
        assertThat(cross.getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        jdbc.update("UPDATE afv_media_resource SET state = 'DELETED', version = version + 1 WHERE id = ?",media);
        var moved = ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.node.layout","nodeId",node,"expectedLayoutVersion",0,
                "layout",Map.of("x",100,"y",100,"width",300,"locked",false,"collapsed",false))))));
        ok(post(f,"/undo",Map.of("requestId",UUID.randomUUID().toString(),"commandId",moved.path("commandId").asText())));
    }

    @Test void createAndPlaceIsOneTransactionAndReplaysExactIdentifiers() {
        var f = fixture();
        String node = UUID.randomUUID().toString();
        var command = placement(node, "scriptEpisode", Map.of("scriptId", f.id(), "parentVersion", 0, "episodeNumber", 1, "title", "新分集"));
        var first = ok(post(f, "/resource-commands/create-and-place", command));
        var repeated = ok(post(f, "/resource-commands/create-and-place", command));
        assertThat(repeated).isEqualTo(first);
        long episode = first.path("resource").path("result").path("id").asLong();
        assertThat(first.path("canvas").path("changes").path("upsertedNodes").get(0).path("resource").path("id").asLong()).isEqualTo(episode);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_script_episode WHERE script_id = ?", Integer.class, f.id())).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_node WHERE id = ? AND canvas_id = ?", Integer.class, node, f.canvas())).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_script WHERE id = ?", Integer.class, f.id())).isEqualTo(1);
    }

    @Test void graphFailureRollsBackCreatedBusinessRowsVersionsAndJournals() {
        var f = fixture();
        String reused = UUID.randomUUID().toString();
        ok(post(f, "/resource-commands/create-and-place", placement(reused, "scriptEpisode", Map.of("scriptId", f.id(), "parentVersion", 0, "episodeNumber", 1))));
        var before = jdbc.queryForObject("SELECT version FROM afv_script WHERE id = ?", Integer.class, f.id());
        var invalid = placement(reused, "scriptEpisode", Map.of("scriptId", f.id(), "parentVersion", before, "episodeNumber", 2));
        var response = post(f, "/resource-commands/create-and-place", invalid);
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(response.getBody().path("data").path("errorCode").asText()).isEqualTo("INVALID_COMMAND");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_script_episode WHERE script_id = ?", Integer.class, f.id())).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_script WHERE id = ?", Integer.class, f.id())).isEqualTo(before);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE request_id = ?", Integer.class, invalid.get("requestId") + ":resource")).isZero();
        assertThat(ok(get(f, "", f.owner())).path("graph").path("nodes")).hasSize(1);
    }

    @Test void collectionPreservesOrderAndVersionAndRequiresCurrentCanvasAndProjectRights() {
        var f = fixture();
        jdbc.update("INSERT INTO afv_script_episode(id, script_id, episode_number, title) VALUES (?, ?, 1, '分集')", f.id(), f.id());
        jdbc.update("INSERT INTO afv_script_scene_item(id, script_id, episode_id, scene_number, sort_order) VALUES (?, ?, ?, '后', 2), (?, ?, ?, '前', 1)", f.id(), f.id(), f.id(), f.id() + 1, f.id(), f.id());
        var collection = ok(get(f, "/collections/script/" + f.id() + "?depth=3", f.owner()));
        assertThat(collection.path("root").path("canEdit").asBoolean()).isTrue();
        assertThat(collection.path("descendants")).hasSize(3);
        assertThat(collection.path("descendants").get(1).path("data").path("sceneNumber").asText()).isEqualTo("前");
        assertThat(collection.path("descendants").get(2).path("data").path("sceneNumber").asText()).isEqualTo("后");
        assertThat(collection.path("descendants").get(1).path("parentType").asText()).isEqualTo("scriptEpisode");
        assertThat(ok(get(f, "/resource-commands/script/" + f.id(), f.owner())).path("canEdit").asBoolean()).isTrue();
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (?, ?, 4)", f.id(), f.id() + 1);
        assertThat(ok(get(f, "/collections/script/" + f.id() + "?depth=3", f.viewer())).path("root").path("canEdit").asBoolean()).isFalse();
        assertThat(ok(get(f, "/resource-commands/script/" + f.id(), f.viewer())).path("canEdit").asBoolean()).isFalse();
        jdbc.update("DELETE FROM afv_project_member WHERE project_id = ?", f.id());
        assertThat(get(f, "/collections/script/" + f.id(), f.viewer()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var another = fixture();
        assertThat(get(f, "/collections/script/" + another.id(), f.owner()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void commentsHistoryAndCheckpointKeepFrozenBusinessBodyThroughHttp() {
        var f = fixture();
        String node = UUID.randomUUID().toString();
        var created = ok(post(f, "/resource-commands/create-and-place", placement(node,"scriptEpisode",Map.of("scriptId",f.id(),"parentVersion",0,"episodeNumber",1,"title","检查点原文"))));
        long episode = created.path("resource").path("result").path("id").asLong();
        while (projection.publishNext()) { /* 先完成业务变化投影，再选择要命名的画布版本。 */ }
        var snapshot = ok(get(f,"",f.owner()));
        String checkpoint = UUID.randomUUID().toString();
        ok(post(f,"/checkpoints",Map.of("requestId",UUID.randomUUID().toString(),"id",checkpoint,"title","修改前","expectedRevision",snapshot.path("document").path("revision").asLong())));
        String comment = UUID.randomUUID().toString();
        ok(post(f,"/comments",Map.of("requestId",UUID.randomUUID().toString(),"id",comment,"nodeId",node,"content","请检查这集")));
        var stored = ok(get(f,"/comments/"+comment,f.owner()));
        assertThat(stored.path("content").asText()).isEqualTo("请检查这集");
        assertThat(stored.path("nodeId").asText()).isEqualTo(node);
        ok(post(f,"/resource-commands",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","scriptEpisode","resourceId",episode,"expectedVersion",0,"patch",Map.of("title","修改后"))));
        var frozen = ok(get(f,"/checkpoints/"+checkpoint,f.owner()));
        assertThat(frozen.toString()).contains("检查点原文").doesNotContain("修改后");
        assertThat(ok(get(f,"/history",f.owner())).toString()).contains("canvas.graph.changed", "canvas.checkpoint.created", "canvas.comment.created");
        var undone = ok(post(f,"/undo",Map.of("requestId",UUID.randomUUID().toString(),"commandId",created.path("canvas").path("commandId").asText())));
        assertThat(undone.path("changes").path("deletedNodeIds").get(0).asText()).isEqualTo(node);
        assertThat(ok(get(f,"/resource-commands/scriptEpisode/"+episode,f.owner())).path("data").path("title").asText()).isEqualTo("修改后");
    }

    private Fixture fixture() {
        long id = IDS.addAndGet(10);
        jdbc.update("INSERT INTO sys_user(id, username, password) VALUES (?, ?, 'test'), (?, ?, 'test')",id,"canvas-owner-"+id,id+1,"canvas-viewer-"+id);
        jdbc.update("INSERT INTO afv_project(id, name, owner_type, owner_id) VALUES (?, '画布测试项目', 1, ?)",id,id);
        jdbc.update("INSERT INTO afv_script(id, project_id, title) VALUES (?, ?, '剧本')",id,id);
        var owner = headers(id,"canvas-owner-"+id);
        var viewer = headers(id+1,"canvas-viewer-"+id);
        String canvas = UUID.randomUUID().toString();
        ok(http.postForEntity("/api/canvases",new HttpEntity<>(Map.of("id",canvas,"title","测试画布","projectId",id),owner),JsonNode.class));
        return new Fixture(id,canvas,owner,viewer);
    }
    private HttpHeaders headers(long id, String name) {
        var headers = new HttpHeaders();
        headers.setBearerAuth(tokens.createToken(id,name,null).getAccessToken());
        headers.setContentType(MediaType.APPLICATION_JSON);
        return headers;
    }
    private Map<String,Object> placement(String node, String type, Map<String,Object> fields) {
        return Map.of("requestId",UUID.randomUUID().toString(),"nodeId",node,"title","业务节点","resourceType",type,"fields",fields,"x",180,"y",240);
    }
    private ResponseEntity<JsonNode> post(Fixture f, String suffix, Object command) {
        return http.postForEntity("/api/canvases/"+f.canvas()+suffix,new HttpEntity<>(command,f.owner()),JsonNode.class);
    }
    private ResponseEntity<JsonNode> get(Fixture f, String suffix, HttpHeaders headers) {
        return http.exchange("/api/canvases/"+f.canvas()+suffix,HttpMethod.GET,new HttpEntity<>(headers),JsonNode.class);
    }
    private JsonNode ok(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull();
        assertThat(response.getBody().path("code").asInt()).as(response.toString()).isZero();
        return response.getBody().path("data");
    }
}
