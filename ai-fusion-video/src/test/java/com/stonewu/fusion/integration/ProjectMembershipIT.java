package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.security.TokenService;
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
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.awaitility.Awaitility.await;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class ProjectMembershipIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test").withCommand("--default-time-zone=+08:00");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "project-members-" + UUID.randomUUID()).toAbsolutePath();
    static final AtomicLong IDS = new AtomicLong(42_000);
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
    record Fixture(long id, String canvas, HttpHeaders owner, HttpHeaders other) { }

    @Test void rolesHaveDifferentBusinessCanvasAndMembershipPermissions() {
        var f = fixture();
        assertThat(get(f, "", f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(ok(get(f, "/members", f.owner())).path("list").get(0).path("protectedOwner").asBoolean()).isTrue();
        assertThat(ok(get(f, "/members/find?username=membership-" + (f.id()+1), f.owner())).path("role").isNull()).isTrue();
        ok(change(f, command(f, 0, 4), f.owner()));
        assertThat(ok(canvas(f, f.other())).path("permissions").path("canEdit").asBoolean()).isFalse();
        assertThat(get(f, "/members/history", f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(get(f, "/members/find?username=membership-"+f.id(), f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(node(f, f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        ok(change(f, command(f, 1, 3), f.owner()));
        ok(node(f, f.other()));
        assertThat(ok(get(f, "/members", f.other())).path("canManage").asBoolean()).isFalse();
        ok(change(f, command(f, 2, 2), f.owner()));
        assertThat(ok(get(f, "/members", f.other())).path("canManage").asBoolean()).isTrue();
        assertThat(ok(get(f, "/members/history", f.other())).path("total").asLong()).isEqualTo(3);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_project WHERE id=?",Integer.class,f.id())).isZero();
    }

    @Test void repeatedConcurrentRequestsCommitOneAuditAndRejectChangedPayload() {
        var f = fixture();
        var command = command(f, 0, 3);
        Instant start = Instant.now().minusSeconds(1);
        var calls = IntStream.range(0, 5).mapToObj(ignored -> CompletableFuture.supplyAsync(() -> ok(change(f, command, f.owner())))).toList();
        var result = calls.getFirst().join();
        calls.forEach(call -> assertThat(call.join()).isEqualTo(result));
        assertThat(Instant.parse(result.path("createdAt").asText())).isBetween(start, Instant.now().plusSeconds(1));
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_project_membership_command WHERE project_id=?",Long.class,f.id())).isEqualTo(1L);
        var changed = new LinkedHashMap<>(command); changed.put("role", 2);
        assertThat(change(f, changed, f.owner()).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(change(f, command(f, 0, 4), f.owner()).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        var audit = ok(get(f,"/members/history",f.owner())).path("list").get(0);
        assertThat(audit.path("actor").path("userId").asLong()).isEqualTo(f.id());
        assertThat(audit.path("member").path("userId").asLong()).isEqualTo(f.id()+1);
        assertThat(audit.path("change").path("previousRole").isNull()).isTrue();
        assertThat(audit.path("change").path("currentRole").asInt()).isEqualTo(3);
    }

    @Test void simultaneousDifferentChoicesHaveOneWinner() {
        var f = fixture();
        var a = CompletableFuture.supplyAsync(() -> change(f, command(f,0,3),f.owner()));
        var b = CompletableFuture.supplyAsync(() -> change(f, command(f,0,4),f.owner()));
        assertThat(List.of(a.join().getStatusCode(),b.join().getStatusCode())).containsExactlyInAnyOrder(HttpStatus.OK,HttpStatus.CONFLICT);
        assertThat(ok(get(f,"/members",f.owner())).path("revision").asLong()).isEqualTo(1);
    }

    @Test void ownerAndDisabledAccountAreProtectedAndRemovalCanBeReaddedWithoutRevivingCanvasGrants() {
        var f = fixture();
        var ownerChange = command(f,0,4); ownerChange.put("userId",f.id());
        assertThat(change(f,ownerChange,f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(change(f,command(f,0,1),f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        jdbc.update("UPDATE sys_user SET status=0 WHERE id=?",f.id()+1);
        assertThat(change(f,command(f,0,3),f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        jdbc.update("UPDATE sys_user SET status=1 WHERE id=?",f.id()+1);
        ok(change(f,command(f,0,3),f.owner()));
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_project_membership_command WHERE project_id=? AND published=FALSE",Long.class,f.id())).isZero());
        var snapshot = ok(canvas(f,f.owner()));
        ok(http.postForEntity("/api/canvases/"+f.canvas()+"/members",new HttpEntity<>(Map.of("requestId",UUID.randomUUID().toString(),
                "expectedRevision",snapshot.path("document").path("revision").asLong(),"userId",f.id()+1,"canEdit",true,"canExecute",true,"canManage",true,"remove",false),f.owner()),JsonNode.class));
        jdbc.update("UPDATE sys_user SET status=0 WHERE id=?",f.id()+1);
        var removed = ok(change(f,command(f,1,null),f.owner()));
        assertThat(removed.path("removedCanvasGrants").asInt()).isEqualTo(1);
        jdbc.update("UPDATE sys_user SET status=1 WHERE id=?",f.id()+1);
        assertThat(canvas(f,f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(node(f,f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        ok(change(f,command(f,2,4),f.owner()));
        assertThat(ok(canvas(f,f.other())).path("permissions").path("canEdit").asBoolean()).isFalse();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_project_member WHERE project_id=? AND user_id=?",Long.class,f.id(),f.id()+1)).isEqualTo(1L);
        assertThat(jdbc.queryForObject("SELECT removed_canvas_grants FROM afv_project_membership_command WHERE id=?",String.class,removed.path("commandId").asText()))
                .contains(f.canvas(),"can_manage","granted_by");
    }

    @Test void durableNotificationUpdatesPermissionsWithoutChangingGraphAndIsPublishedOnce() {
        var f = fixture();
        String id = ok(change(f,command(f,0,3),f.owner())).path("commandId").asText();
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject("SELECT published FROM afv_project_membership_command WHERE id=?",Boolean.class,id)).isTrue());
        assertThat(ok(canvas(f,f.owner())).path("graph").path("nodes")).isEmpty();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_event WHERE canvas_id=? AND event_type='project.members.changed'",Long.class,f.canvas())).isEqualTo(1L);
        assertThat(jdbc.queryForObject("SELECT payload_json FROM afv_canvas_event WHERE canvas_id=? AND event_type='project.members.changed'",String.class,f.canvas()))
                .contains(id,"membershipRevision").doesNotContain("request_hash","removed_canvas_grants");
    }

    @Test void deletedProjectStaleScopeAndUnversionedBodiesDoNotWriteAudit() {
        var f = fixture();
        assertThat(change(f,Map.of("userId",f.id()+1,"role",3),f.owner()).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(change(f,command(f,0,3),f.other()).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        jdbc.update("UPDATE afv_project SET deleted=TRUE WHERE id=?",f.id());
        assertThat(change(f,command(f,0,3),f.owner()).getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_project_membership_command WHERE project_id=?",Long.class,f.id())).isZero();
    }

    private Fixture fixture() {
        long id = IDS.addAndGet(1000);
        for (long user=id;user<=id+1;user++) jdbc.update("INSERT INTO sys_user(id,username,nickname,password) VALUES(?,?,?,'test')",user,"membership-"+user,"成员"+user);
        jdbc.update("INSERT INTO afv_project(id,name,owner_type,owner_id) VALUES(?,'权限审计项目',1,?)",id,id);
        String canvas=UUID.randomUUID().toString();
        var owner=headers(id); var other=headers(id+1);
        ok(http.postForEntity("/api/canvases",new HttpEntity<>(Map.of("id",canvas,"title","权限审计画布","projectId",id),owner),JsonNode.class));
        return new Fixture(id,canvas,owner,other);
    }
    private LinkedHashMap<String,Object> command(Fixture f,long version,Integer role) {
        var result=new LinkedHashMap<String,Object>();
        result.put("requestId",UUID.randomUUID().toString()); result.put("expectedRevision",version);
        result.put("userId",f.id()+1); result.put("role",role); return result;
    }
    private HttpHeaders headers(long id) {
        var headers=new HttpHeaders(); headers.setBearerAuth(tokens.createToken(id,"membership-"+id,null).getAccessToken());
        headers.setContentType(MediaType.APPLICATION_JSON); return headers;
    }
    private ResponseEntity<JsonNode> get(Fixture f,String path,HttpHeaders headers) {
        return http.exchange("/api/project/"+f.id()+path,HttpMethod.GET,new HttpEntity<>(headers),JsonNode.class);
    }
    private ResponseEntity<JsonNode> change(Fixture f,Object command,HttpHeaders headers) {
        return http.postForEntity("/api/project/"+f.id()+"/members",new HttpEntity<>(command,headers),JsonNode.class);
    }
    private ResponseEntity<JsonNode> canvas(Fixture f,HttpHeaders headers) {
        return http.exchange("/api/canvases/"+f.canvas(),HttpMethod.GET,new HttpEntity<>(headers),JsonNode.class);
    }
    private ResponseEntity<JsonNode> node(Fixture f,HttpHeaders headers) {
        var command=Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.node.create","node",
                Map.of("id",UUID.randomUUID().toString(),"kind","text","schemaVersion",1,"title","权限测试",
                        "layout",Map.of("x",20,"y",30,"width",300,"locked",false,"collapsed",false),"config",Map.of("content","内容"),"configVersion",0,"layoutVersion",0))));
        return http.postForEntity("/api/canvases/"+f.canvas()+"/commands",new HttpEntity<>(command,headers),JsonNode.class);
    }
    private JsonNode ok(ResponseEntity<JsonNode> response) {
        assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull();
        assertThat(response.getBody().path("code").asInt()).as(response.toString()).isZero();
        return response.getBody().path("data");
    }
}
