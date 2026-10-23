package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.ai.ToolExecutorRegistry;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScope;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.security.TokenService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Path;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.time.Duration;
import java.util.List;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

import static org.assertj.core.api.Assertions.assertThat;
import static org.awaitility.Awaitility.await;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class CanvasCollaborationIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "canvas-collaboration-" + UUID.randomUUID()).toAbsolutePath();
    static final AtomicLong IDS = new AtomicLong(21_000);
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
    @Autowired StringRedisTemplate redis;
    @Autowired ToolExecutorRegistry tools;
    @Autowired ObjectMapper mapper;
    record Fixture(long id, String canvas, HttpHeaders owner, HttpHeaders other, HttpHeaders third) { }

    @Test void personalSharingSeparatesViewEditExecutionAndManagementAndRevokesEveryEntry() {
        var f = fixture(false);
        assertThat(get(f, "", f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var found = ok(get(f, "/members/find?username=collab-" + (f.id() + 1), f.owner()));
        assertThat(found.path("person").path("userId").asLong()).isEqualTo(f.id() + 1);
        var command = grant(f, false, false, false, false);
        JsonNode first = ok(post(f, "/members", command, f.owner()));
        assertThat(ok(post(f, "/members", command, f.owner()))).isEqualTo(first);
        JsonNode permission = ok(get(f, "", f.other())).path("permissions");
        assertThat(permission.path("canView").asBoolean()).isTrue();
        assertThat(permission.path("canEdit").asBoolean()).isFalse();
        assertThat(permission.path("canExecute").asBoolean()).isFalse();
        assertThat(post(f, "/commands", node("reader"), f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(post(f, "/members", grant(f, true, true, false, false), f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        ok(post(f, "/members", grant(f, false, true, false, false), f.owner()));
        permission = ok(get(f, "", f.other())).path("permissions");
        assertThat(permission.path("canExecute").asBoolean()).isTrue();
        assertThat(permission.path("canEdit").asBoolean()).isFalse();
        ok(post(f, "/members", grant(f, true, false, false, false), f.owner()));
        JsonNode edited = ok(post(f, "/commands", node("member"), f.other()));
        assertThat(edited.path("changes").path("upsertedNodes")).hasSize(1);
        ok(post(f, "/members", grant(f, false, false, false, true), f.owner()));
        assertThat(get(f, "", f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(f, "/events", f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(f, "/history", f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(post(f, "/commands", node("revoked"), f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(ok(get(f, "", f.owner())).path("graph").path("nodes")).hasSize(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_event WHERE canvas_id=? AND event_type='canvas.members.changed'", Long.class, f.canvas())).isEqualTo(4);
    }

    @Test void projectsInheritBusinessRolesAndSeparateCanvasGrantCannotGrantBusinessWrite() {
        var f = fixture(true);
        assertThat(get(f, "/members/find?username=collab-" + (f.id() + 1), f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,4)", f.id(), f.id() + 1);
        assertThat(ok(get(f, "", f.other())).path("permissions").path("canEdit").asBoolean()).isFalse();
        ok(post(f, "/members", grant(f, true, true, false, false), f.owner()));
        ok(post(f, "/commands", node("canvas only"), f.other()));
        assertThat(post(f, "/resource-commands", Map.of("requestId", UUID.randomUUID().toString(), "resourceType", "script",
                "resourceId", f.id(), "expectedVersion", 0, "patch", Map.of("summary", "must not write")), f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        ok(post(f, "/members", grant(f, false, false, false, true), f.owner()));
        assertThat(ok(get(f, "", f.other())).path("permissions").path("canEdit").asBoolean()).isFalse();
        jdbc.update("UPDATE afv_project_member SET role=3 WHERE project_id=? AND user_id=?", f.id(), f.id() + 1);
        assertThat(ok(get(f, "", f.other())).path("permissions").path("canEdit").asBoolean()).isTrue();
        assertThat(ok(get(f, "", f.other())).path("permissions").path("canManage").asBoolean()).isFalse();
        jdbc.update("UPDATE afv_project_member SET deleted=TRUE WHERE project_id=? AND user_id=?", f.id(), f.id() + 1);
        assertThat(get(f, "", f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void creatorDemotionCannotKeepOldManagementAndViewersCannotCreateProjectCanvases() {
        var f = fixture(true);
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,3)", f.id(), f.id() + 1);
        String memberCanvas = UUID.randomUUID().toString();
        ok(http.postForEntity("/api/canvases", new HttpEntity<>(Map.of("id", memberCanvas,"title","成员创建","projectId",f.id()), f.other()), JsonNode.class));
        var memberFixture = new Fixture(f.id(), memberCanvas, f.owner(), f.other(), f.third());
        assertThat(ok(get(memberFixture, "", f.other())).path("permissions").path("canManage").asBoolean()).isTrue();
        jdbc.update("UPDATE afv_project_member SET role=4 WHERE project_id=? AND user_id=?", f.id(), f.id() + 1);
        var permissions = ok(get(memberFixture, "", f.other())).path("permissions");
        assertThat(permissions.path("canManage").asBoolean()).isFalse();
        assertThat(permissions.path("canEdit").asBoolean()).isFalse();
        assertThat(http.postForEntity("/api/canvases", new HttpEntity<>(Map.of("id",UUID.randomUUID().toString(),"title","拒绝","projectId",f.id()),f.other()),JsonNode.class)
                .getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void concurrentMemberChangesConflictAndKeepActorAndPreviousPermissionsInJournal() {
        var f = fixture(false);
        var first = grant(f, true, false, false, false);
        var second = grant(f, false, true, false, false);
        var a = CompletableFuture.supplyAsync(() -> post(f, "/members", first, f.owner()));
        var b = CompletableFuture.supplyAsync(() -> post(f, "/members", second, f.owner()));
        assertThat(List.of(a.join().getStatusCode(), b.join().getStatusCode())).containsExactlyInAnyOrder(HttpStatus.OK, HttpStatus.CONFLICT);
        String audit = jdbc.queryForObject("SELECT request_json FROM afv_canvas_operation WHERE canvas_id=? AND request_id IN (?,?)", String.class,
                f.canvas(), first.get("requestId"), second.get("requestId"));
        assertThat(audit).contains("previous", "current", "userId");
        var changed = new LinkedHashMap<>(first);
        changed.put("canManage", true);
        var original = a.join().getStatusCode().equals(HttpStatus.OK) ? first : second;
        changed.put("requestId", original.get("requestId"));
        assertThat(post(f, "/members", changed, f.owner()).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(ok(get(f, "/members", f.owner())).path("total").asInt()).isEqualTo(2);
    }

    @Test void ownersAreProtectedAndDisabledAccountsCannotBeNewlyInvited() {
        var f = fixture(false);
        var ownerChange = new LinkedHashMap<>(grant(f, false, false, false, true));
        ownerChange.put("userId", f.id());
        assertThat(post(f, "/members", ownerChange, f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        jdbc.update("UPDATE sys_user SET status=0 WHERE id=?", f.id() + 1);
        assertThat(get(f, "/members/find?username=collab-" + (f.id() + 1), f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(post(f, "/members", grant(f, true, false, false, false), f.owner()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(f, "/members", f.third()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void presenceIsSessionScopedExpiresAndRechecksRevokedMembersWithoutChangingTheGraph() {
        var f = fixture(false);
        ok(post(f,"/members",grant(f,true,false,false,false),f.owner()));
        long revision = ok(get(f,"",f.owner())).path("document").path("revision").asLong();
        String ownerSession = UUID.randomUUID().toString(), otherSession = UUID.randomUUID().toString();
        ok(post(f,"/presence",Map.of("sessionId",otherSession,"nodeIds",List.of()),f.other()));
        var present = ok(post(f,"/presence",Map.of("sessionId",ownerSession,"nodeIds",List.of()),f.owner()));
        assertThat(present.path("participants")).hasSize(2);
        assertThat(ok(get(f,"",f.owner())).path("document").path("revision").asLong()).isEqualTo(revision);
        ok(http.exchange("/api/canvases/"+f.canvas()+"/presence/"+otherSession,HttpMethod.DELETE,new HttpEntity<>(f.owner()),JsonNode.class));
        assertThat(ok(post(f,"/presence",Map.of("sessionId",ownerSession,"nodeIds",List.of()),f.owner())).path("participants")).hasSize(2);
        redis.opsForZSet().add("canvas:presence:"+f.canvas(),(f.id()+1)+":"+otherSession,System.currentTimeMillis()-50_000);
        assertThat(ok(post(f,"/presence",Map.of("sessionId",ownerSession,"nodeIds",List.of()),f.owner())).path("participants")).hasSize(1);
        ok(post(f,"/presence",Map.of("sessionId",otherSession,"nodeIds",List.of()),f.other()));
        ok(post(f,"/members",grant(f,false,false,false,true),f.owner()));
        assertThat(ok(post(f,"/presence",Map.of("sessionId",ownerSession,"nodeIds",List.of()),f.owner())).path("participants")).hasSize(1);
        assertThat(post(f,"/presence",Map.of("sessionId",otherSession,"nodeIds",List.of()),f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void businessHistoryHasRealAuthorsAndRestoresOnlyOriginalFieldsWithoutOverwritingAnotherEditorsWork() {
        var f = fixture(true);
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,3)",f.id(),f.id()+1);
        var first = patchScript(f,0,Map.of("summary","初版摘要"),f.owner());
        String command = first.path("commandId").asText();
        var detail = ok(get(f,"/resource-history/"+command,f.other()));
        assertThat(detail.path("summary").path("actorName").asText()).isEqualTo("协作者"+f.id());
        assertThat(detail.path("summary").path("snapshotSchemaVersion").asInt()).isEqualTo(1);
        assertThat(detail.path("revisions")).hasSize(1);
        assertThat(detail.path("revisions").get(0).path("after").path("summary").asText()).isEqualTo("初版摘要");
        JsonNode before = detail.path("revisions").get(0).path("before").path("summary");
        String originalRevision = jdbc.queryForObject("SELECT after_json FROM afv_resource_revision WHERE command_id=?",String.class,command);
        patchScript(f,1,Map.of("genre","悬疑"),f.other());
        var preview = ok(get(f,"/resource-history/"+command+"/restore-preview",f.owner()));
        assertThat(preview.path("canRestore").asBoolean()).isTrue();
        var request = restore(preview);
        var restored = ok(post(f,"/resource-history/restore",request,f.owner()));
        assertThat(ok(post(f,"/resource-history/restore",request,f.owner()))).isEqualTo(restored);
        var current = ok(get(f,"/resource-commands/script/"+f.id(),f.owner())).path("data");
        assertThat(current.path("summary")).isEqualTo(before);
        assertThat(current.path("genre").asText()).isEqualTo("悬疑");
        assertThat(current.path("version").asInt()).isEqualTo(3);
        assertThat(jdbc.queryForObject("SELECT after_json FROM afv_resource_revision WHERE command_id=?",String.class,command)).isEqualTo(originalRevision);
        var page = ok(get(f,"/resource-history?limit=1",f.owner()));
        assertThat(page.path("items")).hasSize(1);
        assertThat(page.path("nextCursor").asText()).isNotEmpty();
        var older = ok(get(f,"/resource-history?limit=1&before="+page.path("nextCursor").asText(),f.owner()));
        assertThat(older.path("items").get(0).path("commandId").asText()).isNotEqualTo(page.path("items").get(0).path("commandId").asText());
    }

    @Test void sameFieldChangesStalePreviewsAndUnauthorizedRestorationAreRejected() {
        var f = fixture(true);
        String command = patchScript(f,0,Map.of("summary","初稿"),f.owner()).path("commandId").asText();
        var preview = ok(get(f,"/resource-history/"+command+"/restore-preview",f.owner()));
        patchScript(f,1,Map.of("summary","后续编辑"),f.owner());
        assertThat(post(f,"/resource-history/restore",restore(preview),f.owner()).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        var conflict = ok(get(f,"/resource-history/"+command+"/restore-preview",f.owner()));
        assertThat(conflict.path("canRestore").asBoolean()).isFalse();
        assertThat(conflict.path("conflicts").get(0).path("field").asText()).isEqualTo("summary");
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,4)",f.id(),f.id()+1);
        ok(get(f,"/resource-history/"+command,f.other()));
        assertThat(get(f,"/resource-history/"+command+"/restore-preview",f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(f,"/resource-history/"+command,f.third()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(post(f,"/resource-history/restore",restore(preview),f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(jdbc.queryForObject("SELECT summary FROM afv_script WHERE id=?",String.class,f.id())).isEqualTo("后续编辑");
    }

    @Test void createdBusinessTreesCanBeConditionallyRemovedAndRestoredWithTheirOriginalIds() {
        var f = fixture(true);
        var created = ok(post(f,"/resource-commands/create",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","asset","projectId",f.id(),
                "fields",Map.of("name","恢复验收角色","type","character","properties",Map.of())),f.owner()));
        String source = created.path("commandId").asText();
        long asset = created.path("result").path("id").asLong();
        assertThat(asset).isPositive();
        var preview = ok(get(f,"/resource-history/"+source+"/restore-preview",f.owner()));
        assertThat(preview.path("changes")).hasSize(2);
        var removed = ok(post(f,"/resource-history/restore",restore(preview),f.owner()));
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_asset WHERE id=?",Boolean.class,asset)).isTrue();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset_item WHERE asset_id=? AND deleted=TRUE",Long.class,asset)).isEqualTo(1);
        var undoRemoval = ok(get(f,"/resource-history/"+removed.path("commandId").asText()+"/restore-preview",f.owner()));
        ok(post(f,"/resource-history/restore",restore(undoRemoval),f.owner()));
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_asset WHERE id=?",Boolean.class,asset)).isFalse();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset_item WHERE asset_id=? AND deleted=FALSE",Long.class,asset)).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_asset WHERE id=?",Integer.class,asset)).isEqualTo(2);
    }

    @Test void legacyAuditIsReadableButMissingRestorationSchemaIsNeverGuessed() {
        var f = fixture(true);
        String source = patchScript(f,0,Map.of("summary","旧格式测试"),f.owner()).path("commandId").asText();
        // 仅在此临时测试库构造旧格式夹具，不回写应用历史。
        jdbc.update("UPDATE afv_resource_command SET request_json=JSON_REMOVE(request_json,'$.schemaVersion') WHERE id=?",source);
        var detail = ok(get(f,"/resource-history/"+source,f.owner()));
        assertThat(detail.path("summary").path("snapshotSchemaVersion").asInt()).isZero();
        assertThat(get(f,"/resource-history/"+source+"/restore-preview",f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(jdbc.queryForObject("SELECT summary FROM afv_script WHERE id=?",String.class,f.id())).isEqualTo("旧格式测试");
    }

    @Test void checkpointRestoresDeletedNodesAndEdgesWithNewVersionsAndCanBeUndone() {
        var f = fixture(false);
        var first = ok(post(f,"/commands",node("版本内节点"),f.owner())).path("changes").path("upsertedNodes").get(0);
        var second = ok(post(f,"/commands",node("连接目标"),f.owner())).path("changes").path("upsertedNodes").get(0);
        String edge = UUID.randomUUID().toString();
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.edge.create","edge",
                Map.of("id",edge,"kind","annotation","sourceNodeId",first.path("id").asText(),"targetNodeId",second.path("id").asText(),"version",0)))),f.owner()));
        String checkpoint = checkpoint(f,"删除之前");
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.node.delete","nodeId",first.path("id").asText(),"expectedConfigVersion",0,"expectedLayoutVersion",0))),f.owner()));
        var added = ok(post(f,"/commands",node("后续新增"),f.owner())).path("changes").path("upsertedNodes").get(0);
        var plan = ok(get(f,"/checkpoints/"+checkpoint+"/restore-preview",f.owner()));
        assertThat(plan.path("nodes")).hasSize(2);
        assertThat(plan.path("upsertedEdges").asInt()).isEqualTo(1);
        var body = checkpointRestore(plan);
        var restored = ok(post(f,"/checkpoints/"+checkpoint+"/restore",body,f.owner()));
        assertThat(ok(post(f,"/checkpoints/"+checkpoint+"/restore",body,f.owner()))).isEqualTo(restored);
        var graph = ok(get(f,"",f.owner())).path("graph");
        assertThat(graph.path("nodes")).hasSize(2);
        assertThat(graph.path("edges")).hasSize(1);
        assertThat(graph.toString()).contains(first.path("id").asText()).doesNotContain(added.path("id").asText());
        assertThat(jdbc.queryForObject("SELECT config_version FROM afv_canvas_node WHERE id=?",Long.class,first.path("id").asText())).isEqualTo(2);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_canvas_edge WHERE id=?",Long.class,edge)).isEqualTo(2);
        ok(post(f,"/undo",Map.of("requestId",UUID.randomUUID().toString(),"commandId",restored.path("commandId").asText()),f.owner()));
        var undone = ok(get(f,"",f.owner())).path("graph");
        assertThat(undone.path("nodes")).hasSize(2);
        assertThat(undone.path("edges")).isEmpty();
        assertThat(undone.toString()).contains(added.path("id").asText()).doesNotContain(first.path("id").asText());
    }

    @Test void checkpointComparisonBecomesStaleAndReadOnlyMembersCannotRestore() {
        var f = fixture(false);
        ok(post(f,"/commands",node("保留"),f.owner()));
        String checkpoint = checkpoint(f,"基础版本");
        ok(post(f,"/commands",node("新增"),f.owner()));
        var plan = ok(get(f,"/checkpoints/"+checkpoint+"/restore-preview",f.owner()));
        ok(post(f,"/commands",node("比较后另一位编辑"),f.owner()));
        assertThat(post(f,"/checkpoints/"+checkpoint+"/restore",checkpointRestore(plan),f.owner()).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        ok(post(f,"/members",grant(f,false,false,false,false),f.owner()));
        assertThat(get(f,"/checkpoints/"+checkpoint+"/restore-preview",f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(post(f,"/checkpoints/"+checkpoint+"/restore",checkpointRestore(plan),f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(ok(get(f,"",f.owner())).path("graph").path("nodes")).hasSize(3);
    }

    @Test void checkpointRestoresGraphWhileKeepingNewBusinessContent() {
        var f = fixture(true);
        String nodeId = UUID.randomUUID().toString();
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.node.create","node",
                Map.of("id",nodeId,"kind","script","schemaVersion",1,"title","旧剧本节点名","config",Map.of(),"configVersion",0,"layoutVersion",0,
                        "resource",Map.of("type","script","id",f.id(),"version",0),"layout",Map.of("x",20,"y",30,"width",300,"locked",false,"collapsed",false))))),f.owner()));
        String checkpoint = checkpoint(f,"剧本早期布局");
        patchScript(f,0,Map.of("summary","后续协作者改好的正文"),f.owner());
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.node.patch","nodeId",nodeId,"expectedConfigVersion",0,"title","新节点名"))),f.owner()));
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT COUNT(*) FROM afv_resource_event WHERE project_id=? AND published=FALSE",Integer.class,f.id())).isZero());
        var plan = ok(get(f,"/checkpoints/"+checkpoint+"/restore-preview",f.owner()));
        assertThat(plan.path("resources")).hasSize(1);
        assertThat(plan.path("resources").get(0).path("currentVersion").asInt()).isEqualTo(1);
        ok(post(f,"/checkpoints/"+checkpoint+"/restore",checkpointRestore(plan),f.owner()));
        assertThat(jdbc.queryForObject("SELECT summary FROM afv_script WHERE id=?",String.class,f.id())).isEqualTo("后续协作者改好的正文");
        assertThat(ok(get(f,"",f.owner())).path("graph").path("nodes").get(0).path("title").asText()).isEqualTo("旧剧本节点名");
        assertThat(ok(get(f,"/checkpoints/"+checkpoint,f.owner())).path("checkpoint").path("authorName").asText()).isEqualTo("协作者"+f.id());
    }

    @Test void checkpointForkUsesNewObjectIdsRecordsLineageAndConcurrentRetriesCreateOnlyOneCanvas() {
        var f = fixture(false);
        var original = ok(post(f,"/commands",node("副本中的内容"),f.owner())).path("changes").path("upsertedNodes").get(0);
        String checkpoint = checkpoint(f,"命名副本来源");
        String destination = UUID.randomUUID().toString();
        var body = Map.of("requestId",UUID.randomUUID().toString(),"canvasId",destination,"title","新的画布副本");
        var calls = new ArrayList<CompletableFuture<JsonNode>>();
        for(int i=0;i<5;i++) calls.add(CompletableFuture.supplyAsync(() -> ok(post(f,"/checkpoints/"+checkpoint+"/fork",body,f.owner()))));
        assertThat(calls.stream().map(call -> call.join().path("id").asText()).distinct()).containsExactly(destination);
        var copied = new Fixture(f.id(),destination,f.owner(),f.other(),f.third());
        var graph = ok(get(copied,"",f.owner())).path("graph");
        assertThat(graph.path("nodes")).hasSize(1);
        assertThat(graph.path("nodes").get(0).path("id").asText()).isNotEqualTo(original.path("id").asText());
        assertThat(graph.path("nodes").get(0).path("configVersion").asInt()).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_operation WHERE canvas_id=?",Integer.class,destination)).isEqualTo(2);
        assertThat(jdbc.queryForObject("SELECT request_json FROM afv_canvas_operation WHERE canvas_id=? AND request_id=?",String.class,destination,body.get("requestId"))).contains(checkpoint,f.canvas());
        var changed = new LinkedHashMap<String,Object>(body); changed.put("title","不能重用旧请求覆盖");
        assertThat(post(f,"/checkpoints/"+checkpoint+"/fork",changed,f.owner()).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(ok(get(f,"",f.owner())).path("graph").path("nodes").get(0).path("id").asText()).isEqualTo(original.path("id").asText());
    }

    @Test void nestedCharacterAndDialogueReferencesPreventRemovalAndUndoOfRestore() {
        var f = fixture(true);
        var created = createCharacter(f);
        long asset = created.path("result").path("id").asLong();
        var preview = ok(get(f,"/resource-history/"+created.path("commandId").asText()+"/restore-preview",f.owner()));
        var removed = ok(post(f,"/resource-history/restore",restore(preview),f.owner()));
        var reactivation = ok(get(f,"/resource-history/"+removed.path("commandId").asText()+"/restore-preview",f.owner()));
        var restored = ok(post(f,"/resource-history/restore",restore(reactivation),f.owner()));
        patchScript(f,0,Map.of("charactersJson",List.of(Map.of("name","阿青","assetId",asset))),f.owner());
        assertThat(get(f,"/resource-history/"+restored.path("commandId").asText()+"/restore-preview",f.owner()).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_asset WHERE id=?",Boolean.class,asset)).isFalse();

        var second = createCharacter(f);
        long dialogueAsset = second.path("result").path("id").asLong();
        jdbc.update("INSERT INTO afv_script_episode(id,script_id,episode_number,title) VALUES(?,?,1,'对白所在分集')",f.id(),f.id());
        jdbc.update("INSERT INTO afv_script_scene_item(id,script_id,episode_id) VALUES(?,?,?)",f.id(),f.id(),f.id());
        ok(post(f,"/resource-commands",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","scriptScene","resourceId",f.id(),"expectedVersion",0,
                "patch",Map.of("dialogues",List.of(Map.of("type",1,"content","灯塔亮了","character_asset_id",dialogueAsset)))),f.owner()));
        assertThat(get(f,"/resource-history/"+second.path("commandId").asText()+"/restore-preview",f.owner()).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_asset WHERE id=?",Boolean.class,dialogueAsset)).isFalse();
    }

    @Test void restoringScriptContentCannotReactivateAnOldRunningTask() {
        var f = fixture(true);
        patchScript(f,0,Map.of("parsingStatus",1,"parsingProgress","旧任务进行中"),f.owner());
        var completed = patchScript(f,1,Map.of("parsingStatus",2,"parsingProgress","已完成","summary","完成时的概要"),f.owner());
        var preview = ok(get(f,"/resource-history/"+completed.path("commandId").asText()+"/restore-preview",f.owner()));
        assertThat(preview.path("canRestore").asBoolean()).isTrue();
        assertThat(preview.path("notices")).hasSize(1);
        assertThat(preview.path("changes").get(0).path("proposed").path("parsingStatus").asInt()).isZero();
        ok(post(f,"/resource-history/restore",restore(preview),f.owner()));
        assertThat(jdbc.queryForObject("SELECT parsing_status FROM afv_script WHERE id=?",Integer.class,f.id())).isZero();
        assertThat(jdbc.queryForObject("SELECT parsing_progress FROM afv_script WHERE id=?",String.class,f.id())).isNull();
    }

    @Test void projectCreationCannotBeUndoneThroughContentRestoreAndSilentlyOrphanTheWorkspace() {
        var f = fixture(false);
        var created = ok(post(f,"/resource-commands/create",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","project","fields",Map.of("name","项目生命周期验收")),f.owner()));
        long project = created.path("result").path("id").asLong();
        String canvas = UUID.randomUUID().toString();
        ok(http.postForEntity("/api/canvases",new HttpEntity<>(Map.of("id",canvas,"title","项目内容恢复","projectId",project),f.owner()),JsonNode.class));
        var target = new Fixture(f.id(),canvas,f.owner(),f.other(),f.third());
        var preview = ok(get(target,"/resource-history/"+created.path("commandId").asText()+"/restore-preview",f.owner()));
        assertThat(preview.path("canRestore").asBoolean()).isFalse();
        assertThat(preview.path("conflicts")).isNotEmpty();
        assertThat(post(target,"/resource-history/restore",restore(preview),f.owner()).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(jdbc.queryForObject("SELECT deleted FROM afv_project WHERE id=?",Boolean.class,project)).isFalse();
    }

    private JsonNode createCharacter(Fixture f) {
        return ok(post(f,"/resource-commands/create",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","asset","projectId",f.id(),
                "fields",Map.of("name","角色"+UUID.randomUUID(),"type","character","properties",Map.of())),f.owner()));
    }

    @Test void checkpointCopiesPinnedTextIntoAProtectedSnapshotAndRunsOnlyTheSelectedDownstreamNode() throws Exception {
        var f=fixture(false);
        String source=UUID.randomUUID().toString(), target=UUID.randomUUID().toString(), edgeId=UUID.randomUUID().toString();
        var edge=new LinkedHashMap<String,Object>(Map.of("id",edgeId,"kind","execution","sourceNodeId",source,"sourcePort","text",
                "targetNodeId",target,"targetPort","texts","inputIndex",0,"version",0));
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(
                Map.of("type","canvas.node.create","node",pinnedNode(source,"text",Map.of("content","固定的旧文案"))),
                Map.of("type","canvas.node.create","node",pinnedNode(target,"textCombine",Map.of())),
                Map.of("type","canvas.edge.create","edge",edge))),f.owner()));
        var run=run(f,"ALL",List.of());
        JsonNode original=null;
        for (var value:run.path("nodes")) if (value.path("nodeId").asText().equals(source)) original=value;
        assertThat(original).isNotNull();
        String outputId=original.at("/outputs/values/text/0/id").asText();
        edge.put("output",Map.of("nodeRunId",original.path("id").asText(),"outputId",outputId));
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(
                Map.of("type","canvas.node.patch","nodeId",source,"expectedConfigVersion",0,"configPatch",Map.of("content","后来改过的文案")),
                Map.of("type","canvas.edge.replace","edge",edge,"expectedVersion",0))),f.owner()));
        String checkpoint=checkpoint(f,"带固定结果的副本");
        String destination=UUID.randomUUID().toString();
        var request=Map.of("requestId",UUID.randomUUID().toString(),"canvasId",destination,"title","保留固定结果");
        ok(post(f,"/checkpoints/"+checkpoint+"/fork",request,f.owner()));
        ok(post(f,"/checkpoints/"+checkpoint+"/fork",request,f.owner()));
        var copied=new Fixture(f.id(),destination,f.owner(),f.other(),f.third());
        var graph=ok(get(copied,"",f.owner())).path("graph");
        var copiedEdge=graph.path("edges").get(0);
        String snapshotId=copiedEdge.path("output").path("snapshotId").asText();
        assertThat(snapshotId).isNotBlank();
        assertThat(copiedEdge.path("output").path("nodeRunId").isNull()).isTrue();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_value_snapshot WHERE canvas_id=?",Integer.class,destination)).isEqualTo(1);
        var result=run(copied,"SINGLE",List.of(copiedEdge.path("targetNodeId").asText()));
        assertThat(result.path("nodes")).hasSize(1);
        assertThat(result.path("nodes").get(0).at("/outputs/values/text/0/text").asText()).isEqualTo("固定的旧文案");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_run WHERE canvas_id=?",Integer.class,f.canvas())).isEqualTo(1);
        assertThat(get(copied,"/value-snapshots/"+snapshotId,f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        ok(post(copied,"/members",grant(copied,false,false,false,false),f.owner()));
        assertThat(ok(get(copied,"/value-snapshots/"+snapshotId,f.other())).path("value").path("text").asText()).isEqualTo("固定的旧文案");
        assertThat(get(f,"",f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(f,"/value-snapshots/"+snapshotId,f.owner()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var context=ToolExecutionContext.forInvocation(f.id(),1,f.id(),ResourceCommandContext.user(new CanvasActor(f.id(),null),destination));
        context.setCanvasScope(new CanvasAgentScope(destination,null,List.of(),List.of(),null,0,0,0));
        var tool=tools.findExecutor("get_canvas_fixed_output");
        assertThat(tool).isNotNull();
        String arguments=mapper.writeValueAsString(Map.of("edgeId",copiedEdge.path("id").asText()));
        var observed=mapper.readTree(tool.execute(arguments,context));
        assertThat(observed.path("status").asText()).isEqualTo("success");
        assertThat(observed.toString()).contains("固定的旧文案");
        context.setCanvasScope(null);
        assertThat(mapper.readTree(tool.execute(arguments,context)).path("status").asText()).isEqualTo("error");
        edge.put("version",1);
        edge.put("output",Map.of("snapshotId",snapshotId,"outputId",outputId));
        assertThat(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(
                Map.of("type","canvas.edge.replace","edge",edge,"expectedVersion",1))),f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }

    private Map<String,Object> pinnedNode(String id,String kind,Map<String,Object> config) {
        return Map.of("id",id,"kind",kind,"schemaVersion",1,"title",kind,"config",config,"configVersion",0,"layoutVersion",0,
                "layout",Map.of("x",20,"y",30,"width",300,"locked",false,"collapsed",false));
    }
    private JsonNode run(Fixture f,String mode,List<String> nodes) {
        var scope=Map.of("userId",f.id());
        var plan=ok(post(f,"/run-plans",Map.of("expectedRevision",ok(get(f,"",f.owner())).path("document").path("revision").asLong(),
                "mode",mode,"nodeIds",nodes,"reuse",Map.of(),"scope",scope,
                "budget",Map.of("maxNodes",10,"maxImages",0,"maxVideoSeconds",0,"maxParallel",2,"maxBatchItems",10)),f.owner()));
        assertThat(plan.path("issues")).isEmpty();
        String id=ok(post(f,"/runs",Map.of("requestId",UUID.randomUUID().toString(),"planId",plan.path("id").asText(),
                "planHash",plan.path("hash").asText(),"scope",scope),f.owner())).path("id").asText();
        await().atMost(Duration.ofSeconds(15)).untilAsserted(() -> assertThat(ok(http.exchange("/api/canvas-runs/"+id,
                HttpMethod.GET,new HttpEntity<>(f.owner()),JsonNode.class)).path("status").asText()).isEqualTo("SUCCEEDED"));
        return ok(http.exchange("/api/canvas-runs/"+id,HttpMethod.GET,new HttpEntity<>(f.owner()),JsonNode.class));
    }

    private String checkpoint(Fixture f,String title) {
        String id=UUID.randomUUID().toString();
        ok(post(f,"/checkpoints",Map.of("requestId",UUID.randomUUID().toString(),"id",id,"title",title,"expectedRevision",ok(get(f,"",f.owner())).path("document").path("revision").asLong()),f.owner()));
        return id;
    }
    private Map<String,Object> checkpointRestore(JsonNode plan) {
        return Map.of("requestId",UUID.randomUUID().toString(),"expectedRevision",plan.path("expectedRevision").asLong(),"digest",plan.path("digest").asText());
    }

    private JsonNode patchScript(Fixture f,int version,Map<String,Object> fields,HttpHeaders actor) {
        return ok(post(f,"/resource-commands",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","script","resourceId",f.id(),
                "expectedVersion",version,"patch",fields),actor));
    }
    private Map<String,Object> restore(JsonNode preview) {
        return Map.of("requestId",UUID.randomUUID().toString(),"commandId",preview.path("commandId").asText(),
                "resourceType",preview.path("resourceType").asText(),"expectedDigest",preview.path("digest").asText());
    }

    @Test void mediaRestorationUpdatesBindingsAndPreservesUnrelatedEditsAndBothFiles() throws Exception {
        var f=fixture(true); String first=restoreImage(f,0x123456),second=restoreImage(f,0x987654);
        long asset=restoreAsset(f);
        assetPatch(f,asset,0,Map.of("coverUrl","media://"+first));
        String command=assetPatch(f,asset,1,Map.of("coverUrl","media://"+second)).path("commandId").asText();
        assetPatch(f,asset,2,Map.of("description","后续协作者的描述"));
        var preview=ok(get(f,"/resource-history/"+command+"/restore-preview",f.owner()));
        var result=ok(post(f,"/resource-history/restore",restore(preview),f.owner()));
        var current=ok(get(f,"/resource-commands/asset/"+asset,f.owner())).path("data");
        assertThat(current.path("coverUrl").asText()).isEqualTo("media://"+first);
        assertThat(current.path("description").asText()).isEqualTo("后续协作者的描述");
        assertThat(current.path("version").asInt()).isEqualTo(4);
        assertThat(jdbc.queryForObject("SELECT media_id FROM afv_media_binding WHERE resource_type='asset' AND resource_id=? AND field_name='coverUrl'",String.class,asset)).isEqualTo(first);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_resource WHERE id IN (?,?) AND state='READY'",Long.class,first,second)).isEqualTo(2);
        var redo=ok(get(f,"/resource-history/"+result.path("commandId").asText()+"/restore-preview",f.owner()));
        ok(post(f,"/resource-history/restore",restore(redo),f.owner()));
        assertThat(jdbc.queryForObject("SELECT cover_url FROM afv_asset WHERE id=?",String.class,asset)).isEqualTo("media://"+second);
    }

    @Test void mediaRevocationAfterRestorePreviewRejectsTheWholeCommandWithoutChangingBusinessData() throws Exception {
        var f=fixture(true); String first=restoreImage(f,0x123456),second=restoreImage(f,0x987654);
        long asset=restoreAsset(f); assetPatch(f,asset,0,Map.of("coverUrl","media://"+first));
        String command=assetPatch(f,asset,1,Map.of("coverUrl","media://"+second)).path("commandId").asText();
        var input=restore(ok(get(f,"/resource-history/"+command+"/restore-preview",f.owner())));
        jdbc.update("UPDATE afv_media_resource SET state='DELETED',version=version+1 WHERE id=?",first);
        assertThat(post(f,"/resource-history/restore",input,f.owner()).getStatusCode()).isEqualTo(HttpStatus.GONE);
        assertThat(get(f,"/resource-history/"+command+"/restore-preview",f.owner()).getStatusCode()).isEqualTo(HttpStatus.GONE);
        assertThat(jdbc.queryForObject("SELECT cover_url FROM afv_asset WHERE id=?",String.class,asset)).isEqualTo("media://"+second);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_asset WHERE id=?",Integer.class,asset)).isEqualTo(2);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE request_id=?",Long.class,input.get("requestId"))).isZero();
    }

    @Test void businessEditorCannotRestoreAnUnboundPrivateFileFromSomeoneElsesHistory() throws Exception {
        var f=fixture(true); String first=restoreImage(f,0x123456),second=restoreImage(f,0x987654);
        long asset=restoreAsset(f); assetPatch(f,asset,0,Map.of("coverUrl","media://"+first));
        String command=assetPatch(f,asset,1,Map.of("coverUrl","media://"+second)).path("commandId").asText();
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,3)",f.id(),f.id()+1);
        ok(get(f,"/resource-history/"+command,f.other()));
        assertThat(http.exchange("/api/media-resources/"+first,HttpMethod.GET,new HttpEntity<>(f.other()),JsonNode.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(f,"/resource-history/"+command+"/restore-preview",f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var request=restore(ok(get(f,"/resource-history/"+command+"/restore-preview",f.owner())));
        assertThat(post(f,"/resource-history/restore",request,f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_asset WHERE id=?",Integer.class,asset)).isEqualTo(2);
    }
    private long restoreAsset(Fixture f) {
        return ok(post(f,"/resource-commands/create",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","asset","projectId",f.id(),
                "fields",Map.of("name","历史媒体恢复","type","character","properties",Map.of())),f.owner())).path("result").path("id").asLong();
    }
    private JsonNode assetPatch(Fixture f,long asset,int version,Map<String,Object> patch) {
        return ok(post(f,"/resource-commands",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","asset","resourceId",asset,"expectedVersion",version,"patch",patch),f.owner()));
    }
    private String restoreImage(Fixture f,int color) throws Exception {
        if(jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config WHERE is_default=TRUE AND deleted=FALSE",Long.class)==0)
            jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('历史媒体测试','local',?,TRUE)",FILES.toString());
        var image=new BufferedImage(4,4,BufferedImage.TYPE_INT_RGB); image.setRGB(1,1,color);
        var bytes=new ByteArrayOutputStream(); ImageIO.write(image,"png",bytes);
        var body=new LinkedMultiValueMap<String,Object>(); body.add("file",new ByteArrayResource(bytes.toByteArray()) { @Override public String getFilename(){return "restore.png";} });
        body.add("requestId",UUID.randomUUID().toString()); var headers=new HttpHeaders(); headers.putAll(f.owner()); headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        return ok(http.postForEntity("/api/media-resources/upload",new HttpEntity<>(body,headers),JsonNode.class)).path("id").asText();
    }

    private Fixture fixture(boolean project) {
        long id = IDS.addAndGet(1000);
        for (long user = id; user <= id + 2; user++) jdbc.update("INSERT INTO sys_user(id,username,nickname,password) VALUES(?,?,?,'test')", user,"collab-"+user,"协作者"+user);
        if (project) {
            jdbc.update("INSERT INTO afv_project(id,name,owner_type,owner_id) VALUES(?,'协作项目',1,?)",id,id);
            jdbc.update("INSERT INTO afv_script(id,project_id,title) VALUES(?,?,'协作剧本')",id,id);
        }
        var owner = headers(id); var other = headers(id + 1); var third = headers(id + 2);
        String canvas = UUID.randomUUID().toString();
        var body = new LinkedHashMap<String,Object>();
        body.put("id",canvas); body.put("title","协作验收"); body.put("projectId",project ? id : null);
        ok(http.postForEntity("/api/canvases",new HttpEntity<>(body,owner),JsonNode.class));
        return new Fixture(id,canvas,owner,other,third);
    }
    private Map<String,Object> grant(Fixture f, boolean edit, boolean execute, boolean manage, boolean remove) {
        return Map.of("requestId",UUID.randomUUID().toString(),"expectedRevision",ok(get(f,"",f.owner())).path("document").path("revision").asLong(),
                "userId",f.id()+1,"canEdit",edit,"canExecute",execute,"canManage",manage,"remove",remove);
    }
    private Map<String,Object> node(String title) {
        return Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.node.create","node",
                Map.of("id",UUID.randomUUID().toString(),"kind","text","schemaVersion",1,"title",title,
                        "layout",Map.of("x",20,"y",30,"width",300,"locked",false,"collapsed",false),
                        "config",Map.of("content","协作内容"),"configVersion",0,"layoutVersion",0))));
    }
    private HttpHeaders headers(long id) {
        var headers = new HttpHeaders(); headers.setBearerAuth(tokens.createToken(id,"collab-"+id,null).getAccessToken());
        headers.setContentType(MediaType.APPLICATION_JSON); return headers;
    }
    private ResponseEntity<JsonNode> get(Fixture f, String path, HttpHeaders headers) {
        return http.exchange("/api/canvases/"+f.canvas()+path,HttpMethod.GET,new HttpEntity<>(headers),JsonNode.class);
    }
    private ResponseEntity<JsonNode> post(Fixture f, String path, Object body, HttpHeaders headers) {
        return http.postForEntity("/api/canvases/"+f.canvas()+path,new HttpEntity<>(body,headers),JsonNode.class);
    }
    private JsonNode ok(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull();
        assertThat(response.getBody().path("code").asInt()).as(response.toString()).isZero();
        return response.getBody().path("data");
    }
}
