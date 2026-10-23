package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import org.junit.jupiter.api.Test;
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
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.util.LinkedMultiValueMap;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import static org.assertj.core.api.Assertions.assertThat;
import static org.awaitility.Awaitility.await;

/** 完整项目复制使用隔离的真实 MySQL / Redis，校验业务关系、密文共享和事务回滚。 */
@SpringBootTest(webEnvironment=SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode=DirtiesContext.ClassMode.AFTER_CLASS)
class CanvasProjectForkIT {
    @Container static final MySQLContainer<?> MYSQL=new MySQLContainer<>("mysql:8.4.6").withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS=new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES=Path.of("target","canvas-project-fork-"+UUID.randomUUID()).toAbsolutePath();
    static final AtomicLong IDS=new AtomicLong(260000);
    @DynamicPropertySource static void properties(DynamicPropertyRegistry properties) {
        properties.add("spring.datasource.url",MYSQL::getJdbcUrl); properties.add("spring.datasource.username",MYSQL::getUsername); properties.add("spring.datasource.password",MYSQL::getPassword);
        properties.add("spring.data.redis.host",REDIS::getHost); properties.add("spring.data.redis.port",()->REDIS.getMappedPort(6379)); properties.add("spring.data.redis.password",()->"");
        properties.add("app.storage.local-base-path",FILES::toString); properties.add("app.agent-workspace.local-base-path",()->FILES.resolve("workspace").toString());
    }
    @Autowired TestRestTemplate http;
    @Autowired TokenService tokens;
    @Autowired JdbcTemplate jdbc;
    record Fixture(long user,Long project,String canvas,HttpHeaders headers) { }

    @Test void draftNodesSurviveTemplateImportAndIndependentCopyWithoutStartingAnything() {
        var source = fixture(true);
        var drafts = Arrays.stream(CanvasNodeKind.values())
                .map(kind -> node(kind.code(), Map.of(), null)).toList();
        create(source, drafts, List.of());

        var template = template(source);
        assertThat(template.path("graph").path("nodes")).hasSize(drafts.size());
        assertThat(template.path("media")).isEmpty();
        assertThat(template.path("providers")).isEmpty();
        var imported = imported(source, ok(templatePost(source, "import", templateRequest(source, template,
                Map.of("media", Map.of(), "providers", Map.of())))));
        var copyReceipt = ok(post(source, "/project-forks", request(source)));
        var copied = imported(source, copyReceipt);

        for (var result : List.of(imported, copied)) {
            var graph = ok(get(result, "")).path("graph");
            assertThat(graph.path("nodes")).hasSize(drafts.size());
            for (var node : graph.path("nodes")) {
                assertThat(node.path("resource").isNull()).isTrue();
                assertThat(node.path("config")).isEmpty();
            }
            assertThat(one("SELECT COUNT(*) FROM afv_canvas_run WHERE canvas_id=?", result.canvas())).isZero();
            assertThat(one("SELECT COUNT(*) FROM afv_media_resource WHERE project_id=?", result.project())).isZero();
        }
        assertThat(ok(get(source, "")).path("graph").path("nodes")).hasSize(drafts.size());
    }

    @Test void independentProjectCopiesEveryRelationshipAndMediaOnceWithAtomicIdempotentReceipts() throws Exception {
        var f=fixture(true); long id=f.user(); long project=f.project();
        String media=upload(f);
        seedStory(f,media);
        create(f,List.of(node("project",Map.of(),ref("project",project)),node("scriptScene",Map.of(),ref("scriptScene",id)),
                node("shot",Map.of(),ref("shot",id)),node("image",Map.of("resourceId",media),null)),List.of());
        var request=request(f);
        var first=CompletableFuture.supplyAsync(()->ok(post(f,"/project-forks",request)));
        var second=CompletableFuture.supplyAsync(()->ok(post(f,"/project-forks",request)));
        JsonNode receipt=first.join(); assertThat(second.join()).isEqualTo(receipt);
        long target=receipt.path("result").path("id").asLong(); String canvas=receipt.path("result").path("canvasId").asText();
        assertThat(target).isNotEqualTo(project);
        long asset=one("SELECT id FROM afv_asset WHERE project_id=?",target);
        long item=one("SELECT id FROM afv_asset_item WHERE asset_id=?",asset);
        long script=one("SELECT id FROM afv_script WHERE project_id=? AND deleted=FALSE",target);
        long scriptEpisode=one("SELECT id FROM afv_script_episode WHERE script_id=?",script);
        long scene=one("SELECT id FROM afv_script_scene_item WHERE script_id=?",script);
        long board=one("SELECT id FROM afv_storyboard WHERE project_id=?",target);
        long boardEpisode=one("SELECT id FROM afv_storyboard_episode WHERE storyboard_id=?",board);
        long boardScene=one("SELECT id FROM afv_storyboard_scene WHERE storyboard_id=?",board);
        long shot=one("SELECT id FROM afv_storyboard_item WHERE storyboard_id=?",board);
        assertThat(List.of(asset,item,script,scriptEpisode,scene,board,boardEpisode,boardScene,shot)).doesNotContain(id);
        assertThat(one("SELECT CAST(JSON_UNQUOTE(JSON_EXTRACT(characters_json,'$[0].assetId')) AS UNSIGNED) FROM afv_script WHERE id=?",script)).isEqualTo(asset);
        assertThat(one("SELECT scene_asset_id FROM afv_script_scene_item WHERE id=?",scene)).isEqualTo(asset);
        assertThat(one("SELECT CAST(JSON_UNQUOTE(JSON_EXTRACT(dialogues,'$[0].character_asset_id')) AS UNSIGNED) FROM afv_script_scene_item WHERE id=?",scene)).isEqualTo(asset);
        assertThat(one("SELECT episode_id FROM afv_script_scene_item WHERE id=?",scene)).isEqualTo(scriptEpisode);
        assertThat(one("SELECT script_episode_id FROM afv_storyboard_episode WHERE id=?",boardEpisode)).isEqualTo(scriptEpisode);
        assertThat(one("SELECT episode_id FROM afv_storyboard_scene WHERE id=?",boardScene)).isEqualTo(boardEpisode);
        assertThat(one("SELECT storyboard_scene_id FROM afv_storyboard_item WHERE id=?",shot)).isEqualTo(boardScene);
        assertThat(one("SELECT scene_asset_item_id FROM afv_storyboard_item WHERE id=?",shot)).isEqualTo(item);
        assertThat(one("SELECT CAST(JSON_UNQUOTE(JSON_EXTRACT(character_ids,'$[0]')) AS UNSIGNED) FROM afv_storyboard_item WHERE id=?",shot)).isEqualTo(item);
        assertThat(one("SELECT version FROM afv_script WHERE id=?",script)).isEqualTo(5);
        assertThat(one("SELECT parsing_status FROM afv_script WHERE id=?",script)).isZero();
        assertThat(one("SELECT compose_status FROM afv_storyboard_episode WHERE id=?",boardEpisode)).isZero();
        String copied=jdbc.queryForObject("SELECT id FROM afv_media_resource WHERE project_id=?",String.class,target);
        assertThat(copied).isNotEqualTo(media);
        assertThat(jdbc.queryForObject("SELECT first_frame_image_url FROM afv_storyboard_item WHERE id=?",String.class,shot)).isEqualTo("media://"+copied);
        assertThat(jdbc.queryForObject("SELECT cover_url FROM afv_asset WHERE id=?",String.class,asset)).isEqualTo("media://"+copied);
        assertThat(one("SELECT COUNT(*) FROM afv_media_copy WHERE media_id=?",copied)).isEqualTo(1);
        assertThat(one("SELECT COUNT(*) FROM afv_resource_revision WHERE command_id=? AND before_json IS NULL",receipt.path("commandId").asText())).isEqualTo(10);
        var copy=new Fixture(f.user(),target,canvas,f.headers());
        var graph=ok(get(copy,""));
        assertThat(graph.path("document").path("projectId").asLong()).isEqualTo(target);
        assertThat(graph.path("graph").path("nodes")).hasSize(4);
        ok(post(copy,"/resource-commands",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","script","resourceId",script,
                "expectedVersion",5,"patch",Map.of("summary","副本单独修改"))));
        assertThat(jdbc.queryForObject("SELECT summary FROM afv_script WHERE project_id=?",String.class,project)).isEqualTo("原项目概要");
        assertThat(one("SELECT version FROM afv_script WHERE project_id=?",project)).isEqualTo(4);
        assertThat(one("SELECT COUNT(*) FROM afv_project_member WHERE project_id=?",target)).isEqualTo(1);
        assertThat(one("SELECT COUNT(*) FROM afv_canvas_run WHERE canvas_id=?",canvas)).isZero();
    }

    @Test void personalCanvasBecomesAProjectWithIndependentAssetsAndMediaWhileTheOriginalStaysPersonal() throws Exception {
        var f=fixture(false); String media=upload(f); long id=f.user();
        jdbc.update("INSERT INTO afv_asset(id,user_id,name,type,owner_type,owner_id,cover_url) VALUES(?,?,'独立角色','character',1,?,?)",id,id,id,"media://"+media);
        jdbc.update("INSERT INTO afv_asset_item(id,asset_id,name,image_url) VALUES(?,?,'角色正面',?)",id,id,"media://"+media);
        create(f,List.of(node("assetItem",Map.of(),ref("assetItem",id))),List.of());
        var result=ok(post(f,"/project-forks",request(f))).path("result");
        long project=result.path("id").asLong();
        assertThat(one("SELECT COUNT(*) FROM afv_script WHERE project_id=?",project)).isEqualTo(1);
        assertThat(one("SELECT COUNT(*) FROM afv_storyboard WHERE project_id=?",project)).isEqualTo(1);
        assertThat(one("SELECT COUNT(*) FROM afv_asset WHERE project_id=?",project)).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT project_id FROM afv_canvas WHERE id=?",Long.class,f.canvas())).isNull();
        assertThat(jdbc.queryForObject("SELECT project_id FROM afv_asset WHERE id=?",Long.class,id)).isNull();
        assertThat(result.path("mediaFiles").asInt()).isEqualTo(1);
    }

    @Test void copyPreviewDetectsBusinessChangesAndRollbackLeavesNoPartialProjectOrMedia() {
        var f=fixture(true);
        create(f,List.of(node("text",Map.of("content","内容"),null)),List.of());
        var request=request(f);
        jdbc.update("UPDATE afv_project SET version=version+1,name='预览后修改' WHERE id=?",f.project());
        long count=one("SELECT COUNT(*) FROM afv_project");
        assertThat(post(f,"/project-forks",request).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(one("SELECT COUNT(*) FROM afv_project")).isEqualTo(count);
        assertThat(one("SELECT COUNT(*) FROM afv_resource_command WHERE request_id=?",request.get("requestId"))).isZero();
        // 模拟曾经有效、后来被移除的模型配置。导出可查看，但新副本必须经过完整图能力校验。
        jdbc.update("UPDATE afv_canvas_node SET kind='textGeneration',config_json=JSON_OBJECT('modelId',999999,'prompt','测试') WHERE canvas_id=?",f.canvas());
        var invalid=request(f);
        assertThat(post(f,"/project-forks",invalid).getStatusCode().is4xxClientError()).isTrue();
        assertThat(one("SELECT COUNT(*) FROM afv_project")).isEqualTo(count);
        assertThat(one("SELECT COUNT(*) FROM afv_canvas WHERE id=?",invalid.get("canvasId"))).isZero();
        assertThat(one("SELECT COUNT(*) FROM afv_resource_command WHERE request_id=?",invalid.get("requestId"))).isZero();
    }

    @Test void sharedPersonalCanvasDoesNotGrantPermissionToCopyPrivateMedia() throws Exception {
        var f=fixture(false); String media=upload(f);
        create(f,List.of(node("image",Map.of("resourceId",media),null)),List.of());
        jdbc.update("INSERT INTO afv_canvas_member(canvas_id,user_id,can_edit,can_execute,can_manage,granted_by) VALUES(?,?,FALSE,FALSE,FALSE,?)",f.canvas(),f.user()+1,f.user());
        var other=new Fixture(f.user()+1,null,f.canvas(),headers(f.user()+1));
        assertThat(get(other,"/project-forks/preview?expectedRevision="+revision(f)).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        var input=request(f);
        assertThat(post(other,"/project-forks",input).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }

    @Test void readOnlyMemberCanCreateAnIndependentProjectButRevocationRejectsAnUnsubmittedPreview() {
        var source=fixture(true);
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,4)",source.project(),source.user()+1);
        var reader=new Fixture(source.user()+1,source.project(),source.canvas(),headers(source.user()+1));
        var first=request(reader);
        var created=ok(post(reader,"/project-forks",first)).path("result");
        assertThat(one("SELECT owner_id FROM afv_project WHERE id=?",created.path("id").asLong())).isEqualTo(reader.user());
        assertThat(one("SELECT owner_id FROM afv_project WHERE id=?",source.project())).isEqualTo(source.user());
        var second=request(reader);
        jdbc.update("UPDATE afv_project_member SET deleted=TRUE WHERE project_id=? AND user_id=?",source.project(),reader.user());
        assertThat(post(reader,"/project-forks",second).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(one("SELECT COUNT(*) FROM afv_canvas WHERE id=?",second.get("canvasId"))).isZero();
        // 已创建副本的幂等回执继续按副本权限读取，无需重新获得原项目权限。
        assertThat(ok(post(reader,"/project-forks",first)).path("result")).isEqualTo(created);
    }

    @Test void archivedAssetDependenciesAreCopiedAsHistoricalRecordsAndNeverReboundByName() throws Exception {
        var f=fixture(true); long id=f.user();
        seedStory(f,upload(f));
        jdbc.update("UPDATE afv_asset SET deleted=TRUE,cover_url=NULL WHERE id=?",id);
        jdbc.update("UPDATE afv_asset_item SET deleted=TRUE,image_url=NULL WHERE asset_id=?",id);
        // 仍有正式场次/镜头引用这些历史编号，副本必须保留归属和删除状态。
        create(f,List.of(node("scriptScene",Map.of(),ref("scriptScene",id))),List.of());
        var preview=ok(get(f,"/project-forks/preview?expectedRevision="+revisionAfterSettled(f)));
        assertThat(preview.path("historicalObjects").asInt()).isEqualTo(2);
        var created=ok(post(f,"/project-forks",request(f))).path("result");
        long project=created.path("id").asLong();
        long asset=one("SELECT id FROM afv_asset WHERE project_id=?",project);
        assertThat(one("SELECT deleted FROM afv_asset WHERE id=?",asset)).isEqualTo(1);
        assertThat(one("SELECT deleted FROM afv_asset_item WHERE asset_id=?",asset)).isEqualTo(1);
        assertThat(one("SELECT scene_asset_id FROM afv_script_scene_item s JOIN afv_script p ON p.id=s.script_id WHERE p.project_id=?",project)).isEqualTo(asset);
    }

    @Test void copiedFixedBusinessResultKeepsOldVersionWhileTheIndependentCurrentObjectCanBeEdited() {
        var f=fixture(true); long project=f.project();
        var source=node("project",Map.of(),ref("project",project));
        var target=node("collection",Map.of("valueType","project"),null);
        var edge=new LinkedHashMap<String,Object>(Map.of("id",UUID.randomUUID().toString(),"kind","execution","sourceNodeId",source.get("id"),"sourcePort","project",
                "targetNodeId",target.get("id"),"targetPort","items","inputIndex",0,"version",0));
        create(f,List.of(source,target),List.of(edge));
        var run=run(f,List.of(),"ALL"); JsonNode original=null;
        for (var value:run.path("nodes")) if (value.path("nodeId").asText().equals(source.get("id"))) original=value;
        assertThat(original).isNotNull();
        edge.put("output",Map.of("nodeRunId",original.path("id").asText(),"outputId",original.at("/outputs/values/project/0/id").asText()));
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.edge.replace","edge",edge,"expectedVersion",0)))));
        ok(post(f,"/resource-commands",Map.of("requestId",UUID.randomUUID().toString(),"resourceType","project","resourceId",project,"expectedVersion",0,"patch",Map.of("description","后来更新"))));
        await().atMost(Duration.ofSeconds(10)).untilAsserted(()->assertThat(one("SELECT COUNT(*) FROM afv_resource_event WHERE project_id=? AND published=FALSE",project)).isZero());
        var result=ok(post(f,"/project-forks",request(f))).path("result");
        var copy=new Fixture(f.user(),result.path("id").asLong(),result.path("canvasId").asText(),f.headers());
        var graph=ok(get(copy,"")).path("graph");
        var fixed=ok(get(copy,"/fixed-outputs/"+graph.path("edges").get(0).path("id").asText()));
        assertThat(fixed.path("resource").path("id").asLong()).isEqualTo(copy.project());
        assertThat(fixed.path("resource").path("version").asLong()).isZero();
        assertThat(one("SELECT version FROM afv_project WHERE id=?",copy.project())).isEqualTo(2);
        assertThat(fixed.path("data").path("description").asText()).isNotEqualTo("后来更新");
        String downstream=null;
        for (var node:graph.path("nodes")) if (node.path("kind").asText().equals("collection")) downstream=node.path("id").asText();
        var executed=run(copy,List.of(downstream),"DEPENDENCIES");
        assertThat(executed.path("nodes")).hasSize(1);
        assertThat(executed.path("nodes").get(0).at("/outputs/values/items/0/resource/id").asLong()).isEqualTo(copy.project());
    }

    private Fixture fixture(boolean project) {
        long user=IDS.addAndGet(1000);
        jdbc.update("INSERT INTO sys_user(id,username,password) VALUES(?,?,'test'),(?,?,'test')",user,"fork-"+user,user+1,"fork-"+(user+1));
        var headers=headers(user); Long id=null;
        if (project) id=ok(http.postForEntity("/api/project",new HttpEntity<>(Map.of("requestId",UUID.randomUUID().toString(),"fields",Map.of("name","副本来源")),headers),JsonNode.class)).path("id").asLong();
        String canvas=UUID.randomUUID().toString(); var body=new HashMap<String,Object>(); body.put("id",canvas); body.put("title","副本测试"); body.put("projectId",id);
        ok(http.postForEntity("/api/canvases",new HttpEntity<>(body,headers),JsonNode.class));
        if (one("SELECT COUNT(*) FROM afv_storage_config WHERE is_default=TRUE AND deleted=FALSE")==0)
            jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('副本测试存储','local',?,TRUE)",FILES.toString());
        return new Fixture(user,id,canvas,headers);
    }

    @Test void portableTemplateCreatesIndependentBusinessObjectsAndMediaWithConcurrentIdempotentImports() throws Exception {
        var f=fixture(true); String media=upload(f); seedStory(f,media);
        create(f,List.of(node("project",Map.of(),ref("project",f.project())),node("shot",Map.of(),ref("shot",f.user())),
                node("image",Map.of("resourceId",media),null)),List.of());
        var template=template(f);
        assertThat(template.toString()).doesNotContain(media,f.canvas(),"ownerId","userId","bgmTaskId");
        assertThat(template.path("objects")).hasSize(10);
        var bindings=mediaBindings(f,template,media);
        var input=templateRequest(f,template,bindings);
        var first=CompletableFuture.supplyAsync(()->ok(templatePost(f,"import",input)));
        var second=CompletableFuture.supplyAsync(()->ok(templatePost(f,"import",input)));
        var receipt=first.join(); assertThat(second.join()).isEqualTo(receipt);
        var copy=imported(f,receipt);
        long asset=one("SELECT id FROM afv_asset WHERE project_id=?",copy.project());
        long item=one("SELECT id FROM afv_asset_item WHERE asset_id=?",asset);
        assertThat(one("SELECT i.scene_asset_item_id FROM afv_storyboard_item i JOIN afv_storyboard b ON b.id=i.storyboard_id WHERE b.project_id=?",copy.project())).isEqualTo(item);
        assertThat(one("SELECT s.scene_asset_id FROM afv_script_scene_item s JOIN afv_script p ON p.id=s.script_id WHERE p.project_id=?",copy.project())).isEqualTo(asset);
        assertThat(one("SELECT version FROM afv_script WHERE project_id=?",copy.project())).isEqualTo(5);
        assertThat(one("SELECT COUNT(*) FROM afv_canvas_run WHERE canvas_id=?",copy.canvas())).isZero();
        assertThat(one("SELECT COUNT(*) FROM afv_resource_revision WHERE command_id=? AND before_json IS NULL",receipt.path("commandId").asText())).isEqualTo(10);
        String alias=jdbc.queryForObject("SELECT id FROM afv_media_resource WHERE project_id=?",String.class,copy.project());
        assertThat(alias).isNotEqualTo(media);
        assertThat(one("SELECT version FROM afv_media_resource WHERE id=?",alias)).isZero();
        var preview=ok(http.postForEntity("/api/media-resources/"+alias+"/preview",new HttpEntity<>(Map.of(),f.headers()),JsonNode.class));
        assertThat(preview.path("sourceVersion").asInt()).isZero();
        await().atMost(Duration.ofSeconds(20)).untilAsserted(() -> assertThat(ok(http.exchange("/api/media-resources/"+alias+"/preview",HttpMethod.GET,
                new HttpEntity<>(f.headers()),JsonNode.class)).path("status").asText()).isEqualTo("SUCCEEDED"));
        jdbc.update("UPDATE afv_media_resource SET state='DELETED',version=version+1 WHERE id=?",media);
        assertThat(ok(templatePost(f,"import",input))).isEqualTo(receipt);
        assertThat(ok(get(copy,"")).path("graph").path("nodes")).hasSize(3);
    }

    @Test void importedFixedTextRunsDownstreamOnlyAndHasExplicitTemplateProvenance() {
        var f=fixture(false); var source=node("text",Map.of("content","灯塔亮起，收到回信。"),null);
        var target=node("textCombine",Map.of("separator","\n"),null);
        var edge=wire(source,"text",target,"texts"); create(f,List.of(source,target),List.of(edge));
        pin(f,edge,source,"text");
        jdbc.update("UPDATE afv_canvas_node SET config_json=JSON_OBJECT('content','后来修改的文案'),config_version=config_version+1 WHERE canvas_id=? AND id=?",f.canvas(),source.get("id"));
        var template=template(f);
        var copy=imported(f,ok(templatePost(f,"import",templateRequest(f,template,Map.of("media",Map.of(),"providers",Map.of())))));
        var graph=ok(get(copy,"")).path("graph");
        String downstream=null; for(var n:graph.path("nodes")) if(n.path("kind").asText().equals("textCombine")) downstream=n.path("id").asText();
        var result=run(copy,List.of(downstream),"DEPENDENCIES");
        assertThat(result.path("nodes")).hasSize(1);
        assertThat(result.path("nodes").get(0).at("/outputs/values/text/0/text").asText()).isEqualTo("灯塔亮起，收到回信。");
        assertThat(jdbc.queryForObject("SELECT source_kind FROM afv_canvas_value_snapshot WHERE canvas_id=?",String.class,copy.canvas())).isEqualTo("TEMPLATE");
        assertThat(jdbc.queryForObject("SELECT source_canvas_id FROM afv_canvas_value_snapshot WHERE canvas_id=?",String.class,copy.canvas())).isNull();
        assertThat(jdbc.queryForObject("SELECT template_hash FROM afv_canvas_value_snapshot WHERE canvas_id=?",String.class,copy.canvas())).matches("[0-9a-f]{64}");
    }

    @Test void replacedTextFileIsReadAgainForImportedFixedResults() throws Exception {
        var f=fixture(false); String original=uploadText(f,"原文本");
        var source=node("textFile",Map.of("resourceId",original),null); var target=node("textCombine",Map.of(),null);
        var edge=wire(source,"media",target,"texts"); create(f,List.of(source,target),List.of(edge)); pin(f,edge,source,"media");
        var template=template(f); String replacement=uploadText(f,"替换后的文本");
        var bindings=mediaBindings(f,template,replacement);
        assertThat(ok(templatePost(f,"preview",Map.of("templateJson",template.toString(),"bindings",bindings))).path("notices")).hasSize(1);
        var copy=imported(f,ok(templatePost(f,"import",templateRequest(f,template,bindings))));
        String edgeId=ok(get(copy,"")).at("/graph/edges/0/id").asText();
        assertThat(ok(get(copy,"/fixed-outputs/"+edgeId)).path("text").asText()).isEqualTo("替换后的文本");
    }

    @Test void templateFilesCannotGrantPrivateMediaPermissionsAndChangedBindingsRejectWithoutSideEffects() throws Exception {
        var f=fixture(false); String media=upload(f); create(f,List.of(node("image",Map.of("resourceId",media),null)),List.of());
        var template=template(f); var bindings=mediaBindings(f,template,media);
        var other=new Fixture(f.user()+1,null,f.canvas(),headers(f.user()+1));
        ok(templatePost(other,"inspect",Map.of("templateJson",template.toString())));
        var request=templateRequest(f,template,bindings); long projects=one("SELECT COUNT(*) FROM afv_project");
        assertThat(templatePost(other,"preview",Map.of("templateJson",template.toString(),"bindings",bindings)).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(templatePost(other,"import",request).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        jdbc.update("UPDATE afv_media_resource SET version=version+1 WHERE id=?",media);
        assertThat(templatePost(f,"import",request).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(one("SELECT COUNT(*) FROM afv_project")).isEqualTo(projects);
        assertThat(one("SELECT COUNT(*) FROM afv_resource_command WHERE request_id=?",request.get("requestId"))).isZero();
    }

    @Test void templatesRejectUnknownFieldsOwnershipAndBrokenReferencesBeforeCreatingAnything() throws Exception {
        var f=fixture(true); seedStory(f,upload(f)); create(f,List.of(node("project",Map.of(),ref("project",f.project()))),List.of());
        var template=template(f); long projects=one("SELECT COUNT(*) FROM afv_project");
        var unknown=template.deepCopy(); unknown.put("unexpected",true);
        assertThat(templatePost(f,"inspect",Map.of("templateJson",unknown.toString())).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        var owner=template.deepCopy(); ((ObjectNode)owner.at("/objects/0/data")).put("ownerId",f.user());
        assertThat(templatePost(f,"inspect",Map.of("templateJson",owner.toString())).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        var broken=template.deepCopy();
        for(var entry:broken.path("objects")) if(entry.path("type").asText().equals("shot")) ((ObjectNode)entry.path("data")).put("sceneAssetItemId",3999);
        assertThat(templatePost(f,"inspect",Map.of("templateJson",broken.toString())).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(one("SELECT COUNT(*) FROM afv_project")).isEqualTo(projects);
    }

    @Test void providerBindingsAreExplicitAndDoNotCarryCredentialsOrStartGeneration() {
        var f=fixture(false); long id=f.user();
        jdbc.update("INSERT INTO afv_api_config(id,name,platform,text_protocol,api_url,api_key) VALUES(?,'模板测试','openai_compatible','openai_compatible','http://127.0.0.1:1','template-secret')",id);
        for(long model:List.of(id,id+1)) jdbc.update("INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels) VALUES(?,'文本绑定',?,1,'openai_compatible',?,'{}',1,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())",model,"template-text-"+model,id);
        create(f,List.of(node("textGeneration",Map.of("modelId",id,"prompt","写一个场景"),null)),List.of());
        var template=template(f); assertThat(template.path("providers")).hasSize(1);
        assertThat(template.at("/graph/nodes/0/config").has("modelId")).isFalse();
        assertThat(template.toString()).doesNotContain("template-secret","api_key","apiConfigId");
        String slot=template.at("/providers/0/id").asText();
        var bindings=Map.of("media",Map.of(),"providers",Map.of(slot,Map.of("id",Long.toString(id+1))));
        var copy=imported(f,ok(templatePost(f,"import",templateRequest(f,template,bindings))));
        assertThat(ok(get(copy,"")).at("/graph/nodes/0/config/modelId").asLong()).isEqualTo(id+1);
        assertThat(one("SELECT COUNT(*) FROM afv_canvas_run WHERE canvas_id=?",copy.canvas())).isZero();
    }

    @Test void templatePreservesIncompatibleInputsAndIncompleteBatchUntilRunPreflight() {
        var source = fixture(true);
        long model = source.user();
        jdbc.update("INSERT INTO afv_api_config(id,name,platform,text_protocol,api_url,api_key) VALUES(?,'草稿模板','openai_compatible','openai_compatible','http://127.0.0.1:1','template-secret')", model);
        jdbc.update("INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels) VALUES(?,'仅文生图','draft-image',2,'openai_compatible',?,'{\"supportReferenceImages\":false}',1,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())", model, model);
        var shot = node("shot", Map.of(), null);
        var image = node("imageGeneration", Map.of("modelId", model, "prompt", "灯塔"), null);
        var batch = node("foreach", Map.of("operationKind", "imageGeneration", "operationConfig", Map.of()), null);
        create(source, List.of(shot, image, batch), List.of(wire(shot, "firstFrame", image, "referenceImages")));
        var template = template(source);
        String slot = template.at("/providers/0/id").asText();
        var bindings = Map.of("media", Map.of(), "providers", Map.of(slot, Map.of("id", Long.toString(model))));
        var copied = imported(source, ok(templatePost(source, "import", templateRequest(source, template, bindings))));
        var graph = ok(get(copied, "")).path("graph");
        assertThat(graph.path("nodes")).hasSize(3);
        assertThat(graph.path("edges")).hasSize(1);
        var plan = ok(post(copied, "/run-plans", Map.of("expectedRevision", revisionAfterSettled(copied),
                "mode", "ALL", "nodeIds", List.of(), "reuse", Map.of(), "scope", Map.of("userId", copied.user()),
                "budget", Map.of("maxNodes", 10, "maxImages", 1, "maxVideoSeconds", 0, "maxParallel", 1, "maxBatchItems", 1))));
        assertThat(plan.path("issues").toString()).contains("参考图片");
        assertThat(one("SELECT COUNT(*) FROM afv_canvas_run WHERE canvas_id=?", copied.canvas())).isZero();
    }

    private ObjectNode template(Fixture f) { settled(f); return (ObjectNode)ok(get(f,"/template?expectedRevision="+revision(f))); }

    @Test void invalidHierarchyRollsBackAllNewProjectObjectsMediaAndReceipts() throws Exception {
        var f=fixture(true); String media=upload(f); seedStory(f,media);
        create(f,List.of(node("project",Map.of(),ref("project",f.project()))),List.of());
        var template=template(f); ObjectNode script=null,episode=null,scene=null;
        for(var entry:template.path("objects")) {
            if(entry.path("type").asText().equals("script")) script=(ObjectNode)entry;
            if(entry.path("type").asText().equals("scriptEpisode")) episode=(ObjectNode)entry;
            if(entry.path("type").asText().equals("scriptScene")) scene=(ObjectNode)entry;
        }
        var extraScript=script.deepCopy(); extraScript.put("id",3000); ((ObjectNode)extraScript.path("data")).put("id",3000);
        var extraEpisode=episode.deepCopy(); extraEpisode.put("id",3001); ((ObjectNode)extraEpisode.path("data")).put("id",3001).put("scriptId",3000);
        ((ArrayNode)template.path("objects")).add(extraScript).add(extraEpisode);
        ((ObjectNode)scene.path("data")).put("episodeId",3001);
        var input=templateRequest(f,template,mediaBindings(f,template,media));
        long projects=one("SELECT COUNT(*) FROM afv_project"),files=one("SELECT COUNT(*) FROM afv_media_resource");
        assertThat(templatePost(f,"import",input).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(one("SELECT COUNT(*) FROM afv_project")).isEqualTo(projects);
        assertThat(one("SELECT COUNT(*) FROM afv_media_resource")).isEqualTo(files);
        assertThat(one("SELECT COUNT(*) FROM afv_canvas WHERE id=?",input.get("canvasId"))).isZero();
        assertThat(one("SELECT COUNT(*) FROM afv_resource_command WHERE request_id=?",input.get("requestId"))).isZero();
    }
    private Map<String,Object> mediaBindings(Fixture f,JsonNode template,String media) {
        int version=ok(http.exchange("/api/media-resources/"+media,HttpMethod.GET,new HttpEntity<>(f.headers()),JsonNode.class)).path("version").asInt();
        return Map.of("media",Map.of(template.at("/media/0/id").asText(),Map.of("id",media,"version",version)),"providers",Map.of());
    }
    private Map<String,Object> templateRequest(Fixture f,JsonNode template,Map<String,?> bindings) {
        var preview=ok(templatePost(f,"preview",Map.of("templateJson",template.toString(),"bindings",bindings)));
        return Map.of("requestId",UUID.randomUUID().toString(),"canvasId",UUID.randomUUID().toString(),"title","模板独立项目",
                "templateJson",template.toString(),"bindings",bindings,"previewHash",preview.path("hash").asText());
    }
    private Fixture imported(Fixture f,JsonNode receipt) { return new Fixture(f.user(),receipt.at("/result/id").asLong(),receipt.at("/result/canvasId").asText(),f.headers()); }
    private ResponseEntity<JsonNode> templatePost(Fixture f,String action,Object body) {
        return http.postForEntity("/api/canvas-templates/"+action,new HttpEntity<>(body,f.headers()),JsonNode.class);
    }
    private Map<String,Object> wire(Map<String,Object> source,String output,Map<String,Object> target,String input) {
        return new LinkedHashMap<>(Map.of("id",UUID.randomUUID().toString(),"kind","execution","sourceNodeId",source.get("id"),"sourcePort",output,
                "targetNodeId",target.get("id"),"targetPort",input,"inputIndex",0,"version",0));
    }
    private void pin(Fixture f,Map<String,Object> edge,Map<String,Object> source,String output) {
        var executed=run(f,List.of(),"ALL");
        for(var node:executed.path("nodes")) if(node.path("nodeId").asText().equals(source.get("id"))) {
            var value=node.at("/outputs/values/"+output+"/0");
            var reference=new LinkedHashMap<String,Object>(Map.of("nodeRunId",node.path("id").asText(),"outputId",value.path("id").asText()));
            if(value.hasNonNull("mediaId")) reference.put("resourceId",value.path("mediaId").asText());
            edge.put("output",reference);
        }
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",List.of(Map.of("type","canvas.edge.replace","edge",edge,"expectedVersion",0)))));
    }
    private String uploadText(Fixture f,String content) {
        var form=new LinkedMultiValueMap<String,Object>(); form.add("file",new ByteArrayResource(content.getBytes(StandardCharsets.UTF_8)) { @Override public String getFilename(){return "template.txt";} });
        form.add("requestId",UUID.randomUUID().toString()); var headers=new HttpHeaders(); headers.putAll(f.headers()); headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        return ok(http.postForEntity("/api/media-resources/upload",new HttpEntity<>(form,headers),JsonNode.class)).path("id").asText();
    }
    private void seedStory(Fixture f,String media) {
        long id=f.user(),project=f.project();
        long script=one("SELECT id FROM afv_script WHERE project_id=?",project),board=one("SELECT id FROM afv_storyboard WHERE project_id=?",project);
        jdbc.update("INSERT INTO afv_asset(id,user_id,project_id,name,type,owner_type,owner_id,cover_url) VALUES(?,?,?,'阿青','character',1,?,?)",id,id,project,id,"media://"+media);
        jdbc.update("INSERT INTO afv_asset_item(id,asset_id,name,image_url) VALUES(?,?,'正面',?)",id,id,"media://"+media);
        jdbc.update("UPDATE afv_script SET summary='原项目概要',characters_json=JSON_ARRAY(JSON_OBJECT('name','阿青','assetId',?)),version=4,parsing_status=1 WHERE id=?",id,script);
        jdbc.update("INSERT INTO afv_script_episode(id,script_id,episode_number,title) VALUES(?,?,1,'第一集')",id,script);
        jdbc.update("INSERT INTO afv_script_scene_item(id,script_id,episode_id,scene_heading,character_asset_ids,scene_asset_id,prop_asset_ids,dialogues) VALUES(?,?,?,'灯塔',JSON_ARRAY(?),?,JSON_ARRAY(?),JSON_ARRAY(JSON_OBJECT('character_asset_id',?,'type',1,'content','收到回信')))",id,script,id,id,id,id,id);
        jdbc.update("INSERT INTO afv_storyboard_episode(id,storyboard_id,script_episode_id,episode_number,title,compose_status) VALUES(?,?,?,1,'分镜第一集',1)",id,board,id);
        jdbc.update("INSERT INTO afv_storyboard_scene(id,storyboard_id,episode_id,scene_heading) VALUES(?,?,?,'灯塔')",id,board,id);
        jdbc.update("INSERT INTO afv_storyboard_item(id,storyboard_id,storyboard_episode_id,storyboard_scene_id,content,character_ids,scene_asset_item_id,prop_ids,first_frame_image_url) VALUES(?,?,?,?,'画面',JSON_ARRAY(?),?,JSON_ARRAY(?),?)",id,board,id,id,id,id,id,"media://"+media);
    }
    private String upload(Fixture f) throws Exception {
        var bytes=new ByteArrayOutputStream(); ImageIO.write(new BufferedImage(4,4,BufferedImage.TYPE_INT_RGB),"png",bytes);
        var form=new LinkedMultiValueMap<String,Object>(); form.add("file",new ByteArrayResource(bytes.toByteArray()) { @Override public String getFilename(){return "copy.png";} });
        form.add("requestId",UUID.randomUUID().toString()); var headers=new HttpHeaders(); headers.putAll(f.headers()); headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        return ok(http.postForEntity("/api/media-resources/upload",new HttpEntity<>(form,headers),JsonNode.class)).path("id").asText();
    }
    private Map<String,Object> node(String kind,Map<String,Object> config,Map<String,Object> resource) {
        var node=new HashMap<String,Object>(); node.put("id",UUID.randomUUID().toString()); node.put("kind",kind); node.put("schemaVersion",1); node.put("title",kind); node.put("config",config); node.put("resource",resource);
        node.put("configVersion",0); node.put("layoutVersion",0); node.put("layout",Map.of("x",40,"y",80,"width",300,"locked",false,"collapsed",false)); return node;
    }
    private Map<String,Object> ref(String type,long id){return Map.of("type",type,"id",id,"version",0);}
    private void create(Fixture f,List<Map<String,Object>> nodes,List<Map<String,Object>> edges) {
        var operations=new ArrayList<Map<String,Object>>(); nodes.forEach(node->operations.add(Map.of("type","canvas.node.create","node",node))); edges.forEach(edge->operations.add(Map.of("type","canvas.edge.create","edge",edge)));
        ok(post(f,"/commands",Map.of("requestId",UUID.randomUUID().toString(),"operations",operations)));
    }
    private Map<String,Object> request(Fixture f) {
        settled(f);
        var preview=ok(get(f,"/project-forks/preview?expectedRevision="+revision(f)));
        assertThat(preview.path("warnings")).isEmpty();
        return Map.of("requestId",UUID.randomUUID().toString(),"sourceCanvasId",f.canvas(),"expectedRevision",preview.path("revision").asLong(),"digest",preview.path("digest").asText(),"canvasId",UUID.randomUUID().toString(),"title","独立项目副本");
    }
    private JsonNode run(Fixture f,List<String> nodes,String mode) {
        settled(f);
        var scope=Map.of("userId",f.user()); var plan=ok(post(f,"/run-plans",Map.of("expectedRevision",revision(f),"mode",mode,"nodeIds",nodes,"reuse",Map.of(),"scope",scope,
                "budget",Map.of("maxNodes",10,"maxImages",0,"maxVideoSeconds",0,"maxParallel",2,"maxBatchItems",10))));
        assertThat(plan.path("issues")).isEmpty(); String id=ok(post(f,"/runs",Map.of("requestId",UUID.randomUUID().toString(),"planId",plan.path("id").asText(),"planHash",plan.path("hash").asText(),"scope",scope))).path("id").asText();
        await().atMost(Duration.ofSeconds(15)).untilAsserted(()->assertThat(ok(http.exchange("/api/canvas-runs/"+id,HttpMethod.GET,new HttpEntity<>(f.headers()),JsonNode.class)).path("status").asText()).isEqualTo("SUCCEEDED"));
        return ok(http.exchange("/api/canvas-runs/"+id,HttpMethod.GET,new HttpEntity<>(f.headers()),JsonNode.class));
    }
    private long one(String query,Object... parameters){return jdbc.queryForObject(query,Long.class,parameters);}
    private void settled(Fixture f) {
        if (f.project()!=null) await().atMost(Duration.ofSeconds(15)).untilAsserted(()->assertThat(one(
                "SELECT COUNT(*) FROM afv_resource_event WHERE project_id=? AND published=FALSE",f.project())).isZero());
    }
    private long revision(Fixture f){return ok(get(f,"")).path("document").path("revision").asLong();}
    private long revisionAfterSettled(Fixture f){settled(f);return revision(f);}
    private HttpHeaders headers(long user){var headers=new HttpHeaders();headers.setBearerAuth(tokens.createToken(user,"fork-"+user,null).getAccessToken());headers.setContentType(MediaType.APPLICATION_JSON);return headers;}
    private ResponseEntity<JsonNode> post(Fixture f,String path,Object body){return http.postForEntity("/api/canvases/"+f.canvas()+path,new HttpEntity<>(body,f.headers()),JsonNode.class);}
    private ResponseEntity<JsonNode> get(Fixture f,String path){return http.exchange("/api/canvases/"+f.canvas()+path,HttpMethod.GET,new HttpEntity<>(f.headers()),JsonNode.class);}
    private JsonNode ok(ResponseEntity<JsonNode> response){assertThat(response.getStatusCode()).as(response.toString()).isEqualTo(HttpStatus.OK);assertThat(response.getBody()).isNotNull();assertThat(response.getBody().path("code").asInt()).as(response.toString()).isZero();return response.getBody().path("data");}
}
