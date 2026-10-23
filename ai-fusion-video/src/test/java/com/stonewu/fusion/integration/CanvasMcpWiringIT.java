package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.ai.agentscope.mcp.AgentScopeMcpRegistry;
import com.stonewu.fusion.service.ai.agentscope.mcp.McpSingleSubmissionHttp;
import com.stonewu.fusion.service.ai.agentscope.mcp.McpWorkflowTool;
import io.agentscope.core.tool.mcp.McpClientBuilder;
import reactor.core.publisher.Mono;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
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
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import static org.assertj.core.api.Assertions.*;
import static org.awaitility.Awaitility.await;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class CanvasMcpWiringIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6").withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final CanvasMcpTestServer REMOTE = new CanvasMcpTestServer();
    static final AtomicLong IDS = new AtomicLong(57000);
    static final Path FILES = Path.of("target", "mcp-workflow-" + UUID.randomUUID()).toAbsolutePath();
    @DynamicPropertySource static void properties(DynamicPropertyRegistry p) {
        p.add("spring.datasource.url", MYSQL::getJdbcUrl); p.add("spring.datasource.username", MYSQL::getUsername); p.add("spring.datasource.password", MYSQL::getPassword);
        p.add("spring.data.redis.host", REDIS::getHost); p.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379)); p.add("spring.data.redis.password", () -> "");
        p.add("app.storage.local-base-path", FILES::toString); p.add("app.agent-workspace.local-base-path", () -> FILES.resolve("workspace").toString());
    }
    @AfterAll static void closeRemote() { REMOTE.close(); }
    @Autowired TestRestTemplate http;
    @Autowired TokenService tokens;
    @Autowired JdbcTemplate jdbc;
    @Autowired ObjectMapper mapper;
    @MockitoBean AgentScopeMcpRegistry registry;
    private final Map<Long, String> configurations = new ConcurrentHashMap<>();
    @BeforeEach void registry() {
        when(registry.workflowTools(eq("canvas_assistant"), anyLong())).thenAnswer(call -> List.of(tool(call.getArgument(1))));
        when(registry.workflowConfigurationHash(eq("canvas_assistant"), anyLong(), eq("inspect_text"))).thenAnswer(call -> configuration(call.getArgument(1)));
        when(registry.openWorkflowTool(eq("canvas_assistant"), anyLong(), eq("inspect_text"))).thenAnswer(call -> {
            long user = call.getArgument(1);
            var client = McpSingleSubmissionHttp.configure(McpClientBuilder.create("本机 MCP").streamableHttpTransport(REMOTE.url()))
                    .initializationTimeout(Duration.ofSeconds(5)).timeout(Duration.ofSeconds(30)).buildAsync().block(Duration.ofSeconds(6));
            client.initialize().then(Mono.defer(client::listTools)).block(Duration.ofSeconds(6));
            return new McpWorkflowTool.Opened(client, true, tool(user));
        });
    }
    private String configuration(long user) { return configurations.getOrDefault(user, "test-config-" + user); }
    private McpWorkflowTool tool(long user) { return McpWorkflowTool.describe("本机 MCP", REMOTE.tool(), configuration(user), mapper); }

    @Test void realHttpToolResultFeedsDownstreamAndDuplicateStartDoesNotCallTwice() {
        var f = fixture(); var contract = contract(f, "READ_ONLY", "READ_ONLY", "", null);
        String text = "完整链路-" + f.user(), source = id(), mcp = id(), next = id();
        var scenario = REMOTE.scenario(text, "success");
        create(f, List.of(node(source, "text", Map.of("content", text)), node(mcp, "mcpTool", config(contract, "TEXT_FIELD", "text", "{}")), node(next, "textCombine", Map.of())),
                List.of(edge(source, "text", mcp, "input", 0), edge(mcp, "text", next, "texts", 0)));
        var plan = plan(f); assertThat(plan.path("issues")).as(plan.toString()).isEmpty();
        assertThat(plan.at("/cost/mcpCalls").asInt()).isEqualTo(1);
        assertThat(plan.path("nodes").toString()).contains("externalOperation", "本机 MCP", "READ_ONLY");
        var request = startRequest(f, plan);
        String run = ok(post(f, "/api/canvases/" + f.canvas() + "/runs", request)).path("id").asText();
        assertThat(ok(post(f, "/api/canvases/" + f.canvas() + "/runs", request)).path("id").asText()).isEqualTo(run);
        var done = awaitState(f, run, "SUCCEEDED");
        assertThat(nodeResult(done, next).at("/outputs/values/text/0/text").asText()).contains(text);
        assertThat(scenario.calls()).hasValue(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_mcp_job WHERE run_id=? AND status='SUCCEEDED' AND permit_held=FALSE AND request_started_at IS NOT NULL AND received_at IS NOT NULL", Integer.class, run)).isEqualTo(1);
    }
    @Test void missingArgumentsCrossAccountAndChangedConfigurationAreRejectedBeforeCalling() {
        var f = fixture(); var contract = contract(f, "READ_ONLY", "READ_ONLY", "", null);
        create(f, List.of(node(id(), "mcpTool", config(contract, "NONE", "", "{}"))), List.of());
        assertThat(plan(f).path("issues").toString()).contains("MCP_INPUT_INVALID");
        var other = fixture();
        create(other, List.of(node(id(), "mcpTool", config(contract, "NONE", "", "{\"text\":\"blocked\"}"))), List.of());
        assertThat(plan(other).path("issues").toString()).contains("当前账号");
        configurations.put(f.user(), "changed");
        assertThat(plan(f).path("issues").toString()).contains("配置已变化");
        verify(registry, never()).openWorkflowTool(anyString(), anyLong(), anyString());
    }
    @Test void keyedContractUsesJobIdentityAndResponseLostDoesNotResubmitHttpBody() {
        var f = fixture(); var contract = contract(f, "WRITE", "KEYED", "requestKey", null);
        String text = "丢回执-" + f.user(), node = id(); var scenario = REMOTE.scenario(text, "drop");
        create(f, List.of(node(node, "mcpTool", config(contract, "NONE", "", json(Map.of("text", text))))), List.of());
        String run = start(f, plan(f));
        var unknown = awaitState(f, run, "UNKNOWN_SUBMISSION");
        var execution = nodeResult(unknown, node);
        String path = detailPath(run, execution);
        var detail = ok(get(f, path));
        assertThat(REMOTE.request(text).at("/params/arguments/requestKey").asText()).isEqualTo(detail.path("id").asText());
        assertThat(REMOTE.request(text).path("params").path("_meta").path("aiFusionVideo.canvasCallId").asText()).isEqualTo(detail.path("id").asText());
        assertThat(scenario.calls()).hasValue(1);
        assertThat(detail.path("remoteCapacityHeld").asBoolean()).isTrue();
        var decision = decision(f, detail, "CONFIRM_FINISHED");
        var receipt = ok(post(f, path + "/review", decision));
        assertThat(ok(post(f, path + "/review", decision))).isEqualTo(receipt);
        awaitState(f, run, "FAILED");
        assertThat(scenario.calls()).hasValue(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_mcp_decision WHERE job_id=?", Integer.class, detail.path("id").asText())).isEqualTo(1);
    }
    @Test void lateReplyFromOldWorkerCannotPublishUntilUserAdoptsOriginalResult() throws Exception {
        var f = fixture(); var contract = contract(f, "READ_ONLY", "READ_ONLY", "", null);
        String text = "迟到回执-" + f.user(), node = id(); var scenario = REMOTE.scenario(text, "hold");
        create(f, List.of(node(node, "mcpTool", config(contract, "NONE", "", json(Map.of("text", text))))), List.of());
        String run = start(f, plan(f));
        try {
            assertThat(scenario.entered().await(20, TimeUnit.SECONDS)).isTrue();
            String job = jdbc.queryForObject("SELECT id FROM afv_canvas_mcp_job WHERE run_id=?", String.class, run);
            jdbc.update("UPDATE afv_canvas_mcp_job SET lease_until=TIMESTAMPADD(SECOND,-1,UTC_TIMESTAMP(6)) WHERE id=?", job);
            var unknown = awaitState(f, run, "UNKNOWN_SUBMISSION");
            scenario.release().countDown();
            await().atMost(Duration.ofSeconds(8)).untilAsserted(() -> assertThat(jdbc.queryForObject("SELECT result_json IS NOT NULL FROM afv_canvas_mcp_job WHERE id=?", Boolean.class, job)).isTrue());
            assertThat(jdbc.queryForObject("SELECT status FROM afv_canvas_mcp_job WHERE id=?", String.class, job)).isEqualTo("UNKNOWN_SUBMISSION");
            String path = detailPath(run, nodeResult(unknown, node));
            var detail = ok(get(f, path));
            assertThat(detail.path("actions").toString()).contains("USE_RECEIVED");
            ok(post(f, path + "/review", decision(f, detail, "USE_RECEIVED")));
            assertThat(nodeResult(awaitState(f, run, "SUCCEEDED"), node).at("/outputs/values/text/0/text").asText()).contains(text);
            assertThat(scenario.calls()).hasValue(1);
        } finally { scenario.release().countDown(); }
    }
    @Test void cancellationRecordsTheResponseWithoutPublishingIt() throws Exception {
        var f = fixture(); var contract = contract(f, "READ_ONLY", "READ_ONLY", "", null);
        String text = "取消-" + f.user(), node = id(); var scenario = REMOTE.scenario(text, "hold");
        create(f, List.of(node(node, "mcpTool", config(contract, "NONE", "", json(Map.of("text", text))))), List.of());
        String run = start(f, plan(f));
        try {
            assertThat(scenario.entered().await(20, TimeUnit.SECONDS)).isTrue();
            ok(post(f, "/api/canvas-runs/" + run + "/cancel", Map.of("scope", scope(f))));
            scenario.release().countDown();
            var cancelled = awaitState(f, run, "CANCELLED");
            assertThat(nodeResult(cancelled, node).at("/outputs/values")).isEmpty();
            assertThat(jdbc.queryForObject("SELECT received_at IS NOT NULL AND permit_held=FALSE FROM afv_canvas_mcp_job WHERE run_id=?", Boolean.class, run)).isTrue();
            assertThat(scenario.calls()).hasValue(1);
        } finally { scenario.release().countDown(); }
    }
    @Test void actualResponseThatViolatesTheUserContractFailsWithoutPublishingOrRetry() {
        var f = fixture();
        var outputSchema = Map.of("type", "object", "properties", Map.of("result", Map.of("type", "string", "minLength", 3000)), "required", List.of("result"));
        var contract = contract(f, "READ_ONLY", "READ_ONLY", "", outputSchema);
        String text = "格式不符-" + f.user(), node = id(); var scenario = REMOTE.scenario(text, "success");
        create(f, List.of(node(node, "mcpTool", config(contract, "NONE", "", json(Map.of("text", text))))), List.of());
        var done = awaitState(f, start(f, plan(f)), "FAILED");
        var result = nodeResult(done, node);
        assertThat(result.at("/outputs/values")).isEmpty();
        var detail = ok(get(f, detailPath(done.path("id").asText(), result)));
        assertThat(detail.at("/received/valid").asBoolean()).isFalse();
        assertThat(detail.at("/received/response/structuredContent/result").asText()).contains(text);
        assertThat(detail.path("remoteCapacityHeld").asBoolean()).isFalse();
        assertThat(scenario.calls()).hasValue(1);
    }

    @Test void batchUsesFrozenOperationAndPreservesTheActualInputOrder() {
        var f = fixture(); var contract = contract(f, "READ_ONLY", "READ_ONLY", "", null);
        String first = "批量首项-" + f.user(), second = "批量末项-" + f.user();
        var firstRemote = REMOTE.scenario(first, "success"); var secondRemote = REMOTE.scenario(second, "success");
        String a = id(), b = id(), collection = id(), batch = id();
        create(f, List.of(node(a, "text", Map.of("content", first)), node(b, "text", Map.of("content", second)),
                node(collection, "collection", Map.of("valueType", "text")), node(batch, "foreach", Map.of(
                        "operationKind", "mcpTool", "operationConfig", config(contract, "TEXT_FIELD", "text", "{}"),
                        "inputPort", "input", "outputPort", "text", "inputType", "text", "resultType", "text", "maxItems", 2))),
                List.of(edge(a, "text", collection, "items", 0), edge(b, "text", collection, "items", 1), edge(collection, "items", batch, "items", 0)));
        var plan = plan(f);
        assertThat(plan.at("/cost/mcpCalls").asInt()).isEqualTo(2);
        var done = awaitState(f, start(f, plan), "SUCCEEDED");
        var results = nodeResult(done, batch).at("/outputs/values/results");
        assertThat(results).hasSize(2);
        assertThat(results.get(0).path("text").asText()).contains(first);
        assertThat(results.get(1).path("text").asText()).contains(second);
        assertThat(firstRemote.calls()).hasValue(1); assertThat(secondRemote.calls()).hasValue(1);
    }

    @Test void queuedCancellationDoesNotInvokeTheToolWhileThePreviousCallOwnsCapacity() throws Exception {
        var f = fixture(); var contract = contract(f, "READ_ONLY", "READ_ONLY", "", null);
        String firstText = "占用-" + f.user(), nextText = "排队取消-" + f.user(), firstNode = id(), nextNode = id();
        var firstRemote = REMOTE.scenario(firstText, "hold"); var nextRemote = REMOTE.scenario(nextText, "success");
        create(f, List.of(node(firstNode, "mcpTool", config(contract, "NONE", "", json(Map.of("text", firstText))))), List.of());
        String firstRun = start(f, plan(f));
        try {
            assertThat(firstRemote.entered().await(20, TimeUnit.SECONDS)).isTrue();
            create(f, List.of(node(nextNode, "mcpTool", config(contract, "NONE", "", json(Map.of("text", nextText))))), List.of());
            var snapshot = ok(get(f, "/api/canvases/" + f.canvas()));
            var nextPlan = ok(post(f, "/api/canvases/" + f.canvas() + "/run-plans", Map.of("expectedRevision", snapshot.at("/document/revision").asLong(),
                    "mode", "SINGLE", "nodeIds", List.of(nextNode), "reuse", Map.of(), "scope", scope(f),
                    "budget", Map.of("maxNodes", 2, "maxImages", 0, "maxVideoSeconds", 0, "maxParallel", 1, "maxBatchItems", 1))));
            String nextRun = start(f, nextPlan);
            await().atMost(Duration.ofSeconds(8)).untilAsserted(() -> assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_mcp_job WHERE run_id=? AND status='READY'", Integer.class, nextRun)).isEqualTo(1));
            ok(post(f, "/api/canvas-runs/" + nextRun + "/cancel", Map.of("scope", scope(f))));
            awaitState(f, nextRun, "CANCELLED");
            assertThat(nextRemote.calls()).hasValue(0);
            firstRemote.release().countDown(); awaitState(f, firstRun, "SUCCEEDED");
            assertThat(firstRemote.calls()).hasValue(1);
        } finally { firstRemote.release().countDown(); }
    }

    @Test void administratorCanOnlySettleAnUncertainCallAfterTheOriginalIdentityLosesPermission() {
        var f = fixture(); var admin = recoveryAdmin(); var contract = contract(f, "READ_ONLY", "READ_ONLY", "", null);
        String text = "管理员核实-" + f.user(), node = id(); var scenario = REMOTE.scenario(text, "drop");
        create(f, List.of(node(node, "mcpTool", config(contract, "NONE", "", json(Map.of("text", text))))), List.of());
        String run = start(f, plan(f));
        var unknown = awaitState(f, run, "UNKNOWN_SUBMISSION");
        String job = nodeResult(unknown, node).path("externalId").asText();
        String endpoint = "/api/system/execution-recovery/MCP/" + job;
        assertThat(get(f, endpoint).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(ok(get(admin, endpoint)).path("canResolve").asBoolean()).isFalse();
        jdbc.update("UPDATE sys_user SET status=0 WHERE id=?", f.user());
        var target = ok(get(admin, endpoint));
        assertThat(target.path("canResolve").asBoolean()).isTrue();
        var decision = Map.of("adminUserId", admin.user(), "requestId", id(), "expectedRevision", target.path("revision").asLong(),
                "expectedFactsHash", target.path("factsHash").asText(), "note", "已在本机 MCP 服务核实原调用结束", "providerFinished", true);
        ok(post(admin, endpoint + "/confirm-finished", decision));
        assertThat(jdbc.queryForObject("SELECT status FROM afv_canvas_mcp_job WHERE id=?", String.class, job)).isEqualTo("CANCELLED");
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_canvas_mcp_job WHERE id=?", Boolean.class, job)).isFalse();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_execution_recovery_decision WHERE execution_kind='MCP' AND job_id=?", Integer.class, job)).isEqualTo(1);
        assertThat(scenario.calls()).hasValue(1);
    }

    record Fixture(long user, String canvas, HttpHeaders headers) { }
    private Fixture recoveryAdmin() {
        long user = IDS.addAndGet(100);
        jdbc.update("INSERT INTO sys_user(id,username,password) VALUES(?,?,'test-only')", user, "mcp-admin-" + user);
        jdbc.update("INSERT INTO sys_user_role(user_id,role_id) SELECT ?,id FROM sys_role WHERE UPPER(code)='ADMIN' AND deleted=FALSE", user);
        var headers = new HttpHeaders();
        headers.setBearerAuth(tokens.createToken(user, "mcp-admin-" + user, null).getAccessToken());
        headers.setContentType(MediaType.APPLICATION_JSON);
        return new Fixture(user, null, headers);
    }

    @Test void portableMcpTemplateUsesTheImportersContractAndExecutesOnlyAfterExplicitRun() {
        var owner=fixture(); var original=contract(owner,"READ_ONLY","READ_ONLY","",null);
        String text="模板 MCP-"+owner.user(); var scenario=REMOTE.scenario(text,"success");
        create(owner,List.of(node(id(),"mcpTool",config(original,"NONE","",json(Map.of("text",text))))),List.of());
        long revision=ok(get(owner,"/api/canvases/"+owner.canvas())).at("/document/revision").asLong();
        var template=ok(get(owner,"/api/canvases/"+owner.canvas()+"/template?expectedRevision="+revision));
        assertThat(template.toString()).doesNotContain(original.path("id").asText(),configuration(owner.user()));
        assertThat(template.path("providers")).hasSize(1);
        var importer=fixture(); var replacement=contract(importer,"READ_ONLY","READ_ONLY","",null);
        String slot=template.at("/providers/0/id").asText();
        var bindings=Map.of("media",Map.of(),"providers",Map.of(slot,Map.of("id",replacement.path("id").asText(),"version",replacement.path("version").asLong())));
        var preview=ok(post(importer,"/api/canvas-templates/preview",Map.of("templateJson",template.toString(),"bindings",bindings)));
        var created=ok(post(importer,"/api/canvas-templates/import",Map.of("requestId",id(),"canvasId",id(),"title","MCP 模板项目",
                "templateJson",template.toString(),"bindings",bindings,"previewHash",preview.path("hash").asText())));
        assertThat(scenario.calls()).hasValue(0);
        var copy=new Fixture(importer.user(),created.at("/result/canvasId").asText(),importer.headers());
        var graph=ok(get(copy,"/api/canvases/"+copy.canvas()));
        assertThat(graph.at("/graph/nodes/0/config/contractId").asText()).isEqualTo(replacement.path("id").asText());
        await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_event WHERE project_id=? AND published=FALSE",Long.class,created.at("/result/id").asLong())).isZero());
        var result=awaitState(copy,start(copy,plan(copy)),"SUCCEEDED");
        assertThat(result.at("/nodes/0/outputs/values/text/0/text").asText()).contains(text);
        assertThat(scenario.calls()).hasValue(1);
        var foreign=Map.of("media",Map.of(),"providers",Map.of(slot,Map.of("id",original.path("id").asText(),"version",original.path("version").asLong())));
        assertThat(post(importer,"/api/canvas-templates/preview",Map.of("templateJson",template.toString(),"bindings",foreign)).getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
        var invalid=template.deepCopy();
        ((ObjectNode)invalid.at("/graph/nodes/0/config")).put("argumentsJson","{\"wrongParameter\":true}");
        assertThat(post(importer,"/api/canvas-templates/preview",Map.of("templateJson",invalid.toString(),"bindings",bindings)).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }
    private Fixture fixture() {
        long user = IDS.addAndGet(100); String canvas = id();
        jdbc.update("INSERT INTO sys_user(id,username,password) VALUES(?,?,'test-only')", user, "mcp-test-" + user);
        jdbc.update("INSERT INTO afv_project(id,name,owner_type,owner_id) VALUES(?,'MCP 验收',1,?)", user, user);
        var headers = new HttpHeaders(); headers.setBearerAuth(tokens.createToken(user, "mcp-test-" + user, null).getAccessToken()); headers.setContentType(MediaType.APPLICATION_JSON);
        var f = new Fixture(user, canvas, headers);
        ok(post(f, "/api/canvases", Map.of("id", canvas, "title", "MCP 验收", "projectId", user))); return f;
    }
    private JsonNode contract(Fixture f, String effect, String idempotency, String parameter, Object output) {
        var tool = tool(f.user());
        var request = new HashMap<String, Object>();
        request.put("name", "本机 MCP 操作"); request.put("toolName", tool.toolName()); request.put("schemaHash", tool.schemaHash()); request.put("configurationHash", tool.configurationHash());
        request.put("effect", effect); request.put("idempotency", idempotency); request.put("idempotencyParameter", parameter);
        request.put("outputMode", "STRUCTURED_JSON"); request.put("outputSchema", output == null ? tool.outputSchema() : output); request.put("timeoutSeconds", 20); request.put("enabled", true);
        return ok(post(f, "/api/canvas-mcp/contracts", request));
    }
    private Map<String, Object> config(JsonNode contract, String mode, String parameter, String args) {
        return Map.of("contractId", contract.path("id").asText(), "contractVersion", contract.path("version").asLong(), "inputMode", mode, "inputParameter", parameter, "argumentsJson", args);
    }
    private Map<String, Object> node(String id, String kind, Map<String, Object> config) {
        return Map.of("id", id, "kind", kind, "title", kind, "schemaVersion", 1, "config", config, "configVersion", 0, "layoutVersion", 0,
                "layout", Map.of("x", 40, "y", 80, "width", 300, "locked", false, "collapsed", false));
    }
    private Map<String, Object> edge(String source, String sourcePort, String target, String targetPort, int index) {
        return Map.of("id", id(), "kind", "execution", "sourceNodeId", source, "sourcePort", sourcePort, "targetNodeId", target, "targetPort", targetPort, "inputIndex", index, "version", 0);
    }
    private void create(Fixture f, List<Map<String, Object>> nodes, List<Map<String, Object>> edges) {
        List<Object> ops = new ArrayList<>(); nodes.forEach(node -> ops.add(Map.of("type", "canvas.node.create", "node", node))); edges.forEach(edge -> ops.add(Map.of("type", "canvas.edge.create", "edge", edge)));
        ok(post(f, "/api/canvases/" + f.canvas() + "/commands", Map.of("requestId", id(), "operations", ops)));
    }
    private JsonNode plan(Fixture f) {
        var snapshot = ok(get(f, "/api/canvases/" + f.canvas()));
        return ok(post(f, "/api/canvases/" + f.canvas() + "/run-plans", Map.of("expectedRevision", snapshot.at("/document/revision").asLong(), "mode", "ALL", "nodeIds", List.of(), "reuse", Map.of(),
                "scope", scope(f), "budget", Map.of("maxNodes", 100, "maxImages", 0, "maxVideoSeconds", 0, "maxParallel", 2, "maxBatchItems", 10))));
    }
    private Map<String, Object> startRequest(Fixture f, JsonNode plan) { return Map.of("requestId", id(), "planId", plan.path("id").asText(), "planHash", plan.path("hash").asText(), "scope", scope(f)); }
    private String start(Fixture f, JsonNode plan) { assertThat(plan.path("issues")).as(plan.toString()).isEmpty(); return ok(post(f, "/api/canvases/" + f.canvas() + "/runs", startRequest(f, plan))).path("id").asText(); }
    private JsonNode awaitState(Fixture f, String run, String state) {
        await().atMost(Duration.ofSeconds(35)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
            var current = ok(get(f, "/api/canvas-runs/" + run)); assertThat(current.path("status").asText()).as(current.toString()).isEqualTo(state);
        }); return ok(get(f, "/api/canvas-runs/" + run));
    }
    private JsonNode nodeResult(JsonNode run, String id) { for (var node : run.path("nodes")) if (node.path("nodeId").asText().equals(id)) return node; throw new AssertionError("missing node"); }
    private String detailPath(String run, JsonNode node) { return "/api/canvas-runs/" + run + "/nodes/" + node.path("id").asText() + "/mcp"; }
    private Map<String, Object> decision(Fixture f, JsonNode detail, String action) { return Map.of("requestId", id(), "expectedRevision", detail.path("revision").asLong(), "action", action, "note", "已核对本机测试服务的原调用记录", "providerFinished", true, "scope", scope(f)); }
    private Map<String, Object> scope(Fixture f) { return Map.of("userId", f.user()); }
    private ResponseEntity<JsonNode> post(Fixture f, String path, Object body) { return http.postForEntity(path, new HttpEntity<>(body, f.headers()), JsonNode.class); }
    private ResponseEntity<JsonNode> get(Fixture f, String path) { return http.exchange(path, HttpMethod.GET, new HttpEntity<>(f.headers()), JsonNode.class); }
    private JsonNode ok(ResponseEntity<JsonNode> result) { assertThat(result.getStatusCode()).as(result.toString()).isEqualTo(HttpStatus.OK); assertThat(result.getBody().path("code").asInt()).as(result.toString()).isZero(); return result.getBody().path("data"); }
    private String id() { return UUID.randomUUID().toString(); }
    private String json(Object value) { try { return mapper.writeValueAsString(value); } catch (Exception failure) { throw new IllegalArgumentException(failure); } }
}
