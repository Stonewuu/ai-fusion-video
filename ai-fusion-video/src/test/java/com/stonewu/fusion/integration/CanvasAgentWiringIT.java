package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.extensions.model.openai.formatter.OpenAIMessageConverter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.security.TokenService;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.ai.ToolExecutorRegistry;
import com.stonewu.fusion.service.ai.ToolPermissionRisk;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentContext;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentMutations;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScope;
import com.stonewu.fusion.service.canvas.agent.CanvasAgentScopeService;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeScope;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeRepository;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeQueries;
import com.stonewu.fusion.controller.ai.vo.ToolConfirmationReqVO;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeToolScope;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceEventProjectionService;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunService;
import com.stonewu.fusion.service.canvas.run.CanvasRunData;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.ai.provider.AiProviderService;
import com.stonewu.fusion.service.ai.agentscope.kernel.AgentKernelKey;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.run.model.AgentModelRequests;
import com.stonewu.fusion.service.ai.run.RunExecutionSupervisor;
import com.stonewu.fusion.service.ai.run.model.StartAgentExecutionCommand;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.AuthenticatedUserContext;
import com.stonewu.fusion.service.canvas.run.text.CanvasTextRepository;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryService;
import com.stonewu.fusion.service.execution.recovery.ExecutionRecoveryData;
import org.springframework.transaction.support.TransactionTemplate;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.model.ChatUsage;
import com.stonewu.fusion.entity.ai.AiModel;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolUseBlock;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.context.bean.override.mockito.MockitoSpyBean;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.CsvSource;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.ai.model.AiModelRequestOptions;
import com.stonewu.fusion.service.ai.agentscope.AgentScopePipelineRunService;
import com.stonewu.fusion.controller.ai.vo.AiChatReqVO;
import com.stonewu.fusion.service.ai.run.AgentConfirmationService;
import com.stonewu.fusion.service.ai.run.DurableAgentWaitingStateService;
import com.stonewu.fusion.service.ai.run.AgentMessageAllocator;
import com.stonewu.fusion.entity.ai.AgentMessage;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.ToolResultBlock;
import java.util.concurrent.atomic.AtomicReference;
import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import io.agentscope.core.message.ImageBlock;
import io.agentscope.core.message.Base64Source;
import io.agentscope.core.message.VideoBlock;
import io.agentscope.core.message.AudioBlock;
import io.agentscope.core.message.DataBlock;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.Base64;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.MySQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.UUID;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.CompletableFuture;
import java.time.Duration;
import java.time.Instant;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.awaitility.Awaitility.await;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@Testcontainers
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class CanvasAgentWiringIT {
    @Container static final MySQLContainer<?> MYSQL = new MySQLContainer<>("mysql:8.4.6")
            .withDatabaseName("ai_fusion_video").withUsername("afv").withPassword("afv-test");
    @Container static final GenericContainer<?> REDIS = new GenericContainer<>("redis:7.4.5-alpine").withExposedPorts(6379);
    static final Path FILES = Path.of("target", "canvas-agent-" + UUID.randomUUID()).toAbsolutePath();
    static final AtomicLong IDS = new AtomicLong(47000);
    @DynamicPropertySource static void properties(DynamicPropertyRegistry p) {
        p.add("spring.datasource.url", MYSQL::getJdbcUrl); p.add("spring.datasource.username", MYSQL::getUsername); p.add("spring.datasource.password", MYSQL::getPassword);
        p.add("spring.data.redis.host", REDIS::getHost); p.add("spring.data.redis.port", () -> REDIS.getMappedPort(6379)); p.add("spring.data.redis.password", () -> "");
        p.add("app.storage.local-base-path", FILES::toString); p.add("app.agent-workspace.local-base-path", () -> FILES.resolve("workspace").toString());
    }
    @Autowired TestRestTemplate http;
    @Autowired TokenService tokens;
    @Autowired JdbcTemplate jdbc;
    @Autowired AiModelService modelService;
    @Autowired AgentModelRequests modelRequests;
    @Autowired CanvasTextRepository textJobs;
    @Autowired TransactionTemplate transactions;
    @Autowired ExecutionRecoveryService recovery;
    @Autowired CanvasAgentNodeQueries agentDetails;
    @Autowired ResourceEventProjectionService resourceEvents;
    @MockitoSpyBean RunExecutionSupervisor supervisor;
    @Autowired ObjectMapper mapper;
    @Autowired CanvasAgentScopeService scopes;
    @Autowired CanvasAgentMutations mutations;
    @Autowired CanvasQueryService canvases;
    @Autowired ToolExecutorRegistry tools;
    @Autowired ResourceCommandService resources;
    @Autowired CanvasAgentNodeRepository nodeJobs;
    @Autowired CanvasAgentNodeToolScope nodeTools;
    @Autowired CanvasRunService canvasRuns;
    @Autowired MediaIntakeService intake;
    @Autowired AgentScopePipelineRunService pipeline;
    @Autowired AgentConfirmationService confirmations;
    @Autowired DurableAgentWaitingStateService waitingStates;
    @Autowired AgentMessageAllocator messageAllocator;
    @MockitoBean AiProviderService providers;
    record Fixture(long user, long project, String canvas, CanvasAgentScope scope, String runId) { }

    @Test void idsReceiptsAndAuditSurviveRepeatedAgentCallsAndStaleWorkersAreFenced() throws Exception {
        var f = fixture();
        var context = invocation(f, "create-1");
        var input = nodes("$one", "画布上的真实节点");
        var first = mapper.valueToTree(mutations.apply(input, context));
        var replay = mapper.valueToTree(mutations.apply(input, context));
        assertThat(replay).isEqualTo(first);
        String node = first.path("idMappings").path("$one").asText();
        assertThat(canvases.snapshot(f.canvas(), actor(f)).graph().nodes()).singleElement().satisfies(value -> assertThat(value.id()).isEqualTo(node));
        assertThat(jdbc.queryForMap("SELECT origin, agent_run_id, tool_call_id FROM afv_canvas_operation WHERE id=?", first.path("commandId").asText()))
                .containsEntry("origin", "agent").containsEntry("agent_run_id", f.runId()).containsEntry("tool_call_id", "create-1");
        jdbc.update("UPDATE afv_agent_run SET owner_epoch=2 WHERE run_id=?", f.runId());
        assertThatThrownBy(() -> mutations.apply(nodes("$two", "旧工作者不能提交"), invocation(f, "create-2"))).hasMessageContaining("租约");
        assertThat(canvases.snapshot(f.canvas(), actor(f)).graph().nodes()).hasSize(1);
    }

    @Test void failedBatchRollsBackAndChangingRequestContentCannotReplayAnotherReceipt() throws Exception {
        var f = fixture();
        var input = nodes("$one", "第一节点");
        input.addObject().put("type", "canvas.node.delete").put("nodeId", UUID.randomUUID().toString()).put("expectedConfigVersion", 0).put("expectedLayoutVersion", 0);
        assertThatThrownBy(() -> mutations.apply(input, invocation(f, "bad-batch"))).isInstanceOf(RuntimeException.class);
        assertThat(canvases.snapshot(f.canvas(), actor(f)).graph().nodes()).isEmpty();
        mutations.apply(nodes("$one", "一次提交"), invocation(f, "stable-id"));
        assertThatThrownBy(() -> mutations.apply(nodes("$one", "篡改重放"), invocation(f, "stable-id"))).hasMessageContaining("同一请求");
        assertThat(canvases.snapshot(f.canvas(), actor(f)).graph().nodes()).singleElement().satisfies(node -> assertThat(node.title()).isEqualTo("一次提交"));
    }

    @Test void persistedCanvasScopeCannotBeReboundEvenWhenTheUserOwnsBothCanvases() throws Exception {
        var f = fixture();
        String other = canvas(f.user(), f.project());
        var scope = scopes.capture(new CanvasAgentContext(other, List.of(), List.of(), null, 1L), f.project(), actor(f));
        var forged = ToolExecutionContext.forInvocation(f.user(), 1, f.user(),
                new ResourceCommandContext(actor(f), "agent", other, f.runId(), "cross-canvas", "agent-test", 1L));
        forged.setCanvasScope(scope);
        assertThatThrownBy(() -> mutations.apply(nodes("$one", "不能跨画布"), forged)).hasMessageContaining("范围");
        assertThat(canvases.snapshot(other, actor(f)).graph().nodes()).isEmpty();
    }

    @Test void captureChecksProjectSelectionAndCurrentPermissionAndToolRequiresBoundContext() throws Exception {
        var f = fixture();
        assertThatThrownBy(() -> scopes.capture(new CanvasAgentContext(f.canvas(), List.of(), List.of(), null, 1L), f.project() + 1, actor(f)))
                .hasMessageContaining("项目");
        assertThatThrownBy(() -> scopes.capture(new CanvasAgentContext(f.canvas(), List.of(UUID.randomUUID().toString()), List.of(), null, 1L), f.project(), actor(f)))
                .hasMessageContaining("选中节点");
        var context = invocation(f, "read");
        JsonNode result = mapper.readTree(tools.findExecutor("get_canvas_context").execute("{}", context));
        assertThat(result.path("status").asText()).isEqualTo("success");
        assertThat(result.path("scope").path("canvasId").asText()).isEqualTo(f.canvas());
        context.setCanvasScope(null);
        assertThat(mapper.readTree(tools.findExecutor("get_canvas_context").execute("{}", context)).path("status").asText()).isEqualTo("error");
        jdbc.update("UPDATE sys_user SET status=0 WHERE id=?", f.user());
        assertThatThrownBy(() -> scopes.capture(new CanvasAgentContext(f.canvas(), List.of(), List.of(), null, 1L), f.project(), actor(f)))
                .hasMessageContaining("无权");
    }

    @Test void businessCreateAndPlaceUsesAgentAuditAndCurrentVersions() throws Exception {
        var f = fixture();
        var created = mapper.valueToTree(mutations.edit("create", null, "scriptEpisode", "新集", null, null,
                mapper.valueToTree(Map.of("scriptId", f.project(), "parentVersion", 0, "episodeNumber", 1, "title", "第一集")), 60, 80, invocation(f, "business-create")));
        String node = created.path("affectedNodes").get(0).path("id").asText();
        var updated = mapper.valueToTree(mutations.edit("patch", node, null, null, 0, null,
                mapper.valueToTree(Map.of("title", "修订后的第一集")), 0, 0, invocation(f, "business-patch")));
        assertThat(updated.path("status").asText()).isEqualTo("success");
        assertThatThrownBy(() -> mutations.edit("patch", node, null, null, 0, null,
                mapper.valueToTree(Map.of("title", "旧版本覆盖")), 0, 0, invocation(f, "stale-patch")))
                .isInstanceOfSatisfying(CanvasException.class, error -> assertThat(error.getErrorCode()).isEqualTo(CanvasErrorCode.VERSION_CONFLICT));
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE agent_run_id=? AND origin='agent'", Long.class, f.runId())).isEqualTo(2L);
    }

    @Test void sensitiveOperationsStayHighRiskEvenInsideGenericTools() {
        assertThat(tools.findExecutor("apply_canvas_operations").mayRequireHighRiskApproval()).isTrue();
        assertThat(tools.findExecutor("apply_canvas_operations").getPermissionRisk(Map.of("operations", List.of(Map.of("type", "canvas.node.delete")))))
                .isEqualTo(ToolPermissionRisk.HIGH_RISK);
        assertThat(tools.findExecutor("edit_canvas_resource").getPermissionRisk(Map.of("action", "delete"))).isEqualTo(ToolPermissionRisk.HIGH_RISK);
        assertThat(tools.findExecutor("run_canvas_nodes").getPermissionRisk(Map.of())).isEqualTo(ToolPermissionRisk.HIGH_RISK);
        assertThat(tools.findExecutor("confirm_canvas_writeback").getPermissionRisk(Map.of("action", "APPROVE", "acceptReset", false)))
                .isEqualTo(ToolPermissionRisk.HIGH_RISK);
    }

    @Test void nestedToolArgumentsRejectWrongTypesUnknownEnumsAndExcessItems() throws Exception {
        var f = fixture();
        var result = mapper.readTree(tools.findExecutor("validate_canvas_run").execute("""
                {"expectedRevision":1,"mode":"INVENTED","nodeIds":[],"budget":{"maxNodes":1,"maxImages":0,"maxVideoSeconds":0,"maxParallel":1,"maxBatchItems":1}}
                """, invocation(f, "enum")));
        assertThat(result.path("code").asInt()).isEqualTo(400);
        assertThat(result.path("message").asText()).contains("选项");
        result = mapper.readTree(tools.findExecutor("confirm_canvas_writeback").execute("""
                {"canvasRunId":"x","nodeRunId":"y","expectedRevision":1,"proposalHash":"z","action":"APPROVE","note":"test","acceptReset":"true"}
                """, invocation(f, "wrong-type")));
        assertThat(result.path("code").asInt()).isEqualTo(400);
        assertThat(result.path("message").asText()).contains("参数类型");
    }

    @ParameterizedTest(name = "{0}: {1} input, real tool round trip and typed result")
    @CsvSource({
            "ai_media,text,text,get_project",
            "concept_visualizer,text,text,none",
            "story_to_script,text,text,get_project_script",
            "script_full_parse,script,script,get_project_script",
            "script_episode_parse,scriptEpisode,episode,get_script_episode",
            "script_assistant,scriptScene,scene,get_script_episode",
            "episode_script_creator,scriptEpisode,episode,get_script_episode",
            "episode_scene_writer,scriptEpisode,episode,get_script_episode",
            "script_to_storyboard,script,script,get_script_structure",
            "episode_storyboard_writer,scriptEpisode,episode,get_script_episode",
            "storyboard_asset_preprocessor,scriptEpisode,episode,get_script_episode",
            "canvas_asset_extractor,scriptScene,scene,get_script_episode",
            "asset_image_gen,assetItem,assetItem,get_project",
            "asset_image_executor,assetItem,assetItem,get_project",
            "storyboard_frame_gen,shot,shot,get_project",
            "storyboard_frame_executor,shot,shot,get_project",
            "storyboard_video_gen,shot,shot,get_project",
            "storyboard_video_executor,shot,shot,get_project"
    })
    void professionalProfilesTraverseActualSdkWithTypedInputsAndRealTools(
            String profile, String inputType, String sourcePort, String queryTool) throws Exception {
        var f = fixture();
        professionalInputTree(f);
        boolean hasResource = !inputType.equals("text");
        boolean hasTool = !queryTool.equals("none");
        var arguments = Map.<String, Object>of(queryTool.equals("get_script_episode") ? "scriptEpisodeId" : "projectId", f.project());
        var calls = new AtomicInteger();
        when(providers.agentScopeModelFingerprint(any(AiModel.class))).thenAnswer(call -> AgentKernelKey.modelRecordFingerprint(call.getArgument(0)));
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                new ChatModelBase() {
                    @Override public String getModelName() { return "canvas-contract-model"; }
                    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> toolSchemas, GenerateOptions options) {
                        int index = calls.getAndIncrement();
                        var encoded = new CanvasJson(mapper).write(messages);
                        assertThat(encoded).contains(profile).doesNotContain("{projectId}", "{scriptId}", "{scriptEpisodeId}", "{storyboardId}");
                        assertThat(toolSchemas).noneMatch(tool -> Set.of("apply_canvas_operations", "run_canvas_nodes", "save_project", "delete_project", "execute_shell").contains(tool.getName()));
                        if (!hasTool) assertThat(toolSchemas).isEmpty();
                        else if (index == 0) {
                            assertThat(toolSchemas).anyMatch(tool -> tool.getName().equals(queryTool));
                            return Flux.just(ChatResponse.builder().id("profile-query").content(List.of(
                                    ToolUseBlock.builder().id("read-confirmed-input").name(queryTool)
                                            .input(arguments).content(new CanvasJson(mapper).write(arguments)).build()))
                                    .finishReason("tool_calls").build());
                        } else {
                            var results = messages.stream().filter(message -> message.getRole() == MsgRole.TOOL).toList();
                            assertThat(results).isNotEmpty();
                            assertThat(new CanvasJson(mapper).write(results)).contains("success", String.valueOf(f.project()));
                        }
                        return Flux.just(ChatResponse.builder().id("profile-result").content(List.of(TextBlock.builder()
                                .text("已核对专业节点输入：" + profile + "，雨夜古城").build())).finishReason("stop").build());
                    }
                }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        jdbc.update("INSERT INTO afv_api_config(id,name,platform,text_protocol,api_url,api_key) VALUES(?,'SDK 契约测试','openai_compatible','openai_compatible','http://127.0.0.1:1','test-key')", f.project());
        jdbc.update("""
                INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels)
                VALUES(?,'SDK 契约模型','canvas-contract',1,'openai_compatible',?,'{}',2,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())
                """, f.project(), f.project());
        var input = nodes("$agent", "专业能力 " + profile);
        var node = (ObjectNode) input.get(0).path("node");
        node.put("kind", "agent").set("config", mapper.valueToTree(Map.of("agentType", profile, "modelId", f.project(),
                "inputType", inputType, "resultType", "text", "instruction", "核对输入的雨夜古城资料，只读取内容并给出总结，不修改或生成素材",
                "maxAgentRuns", 1, "maxImages", 0, "maxVideoSeconds", 0)));
        if (hasResource) {
            var sourceCommand = nodes("$source", "范围内原始资料");
            var source = (ObjectNode) sourceCommand.get(0).path("node");
            source.put("kind", inputType).set("config", mapper.createObjectNode());
            source.set("resource", mapper.valueToTree(Map.of("type", inputType, "id", f.project(), "version", 0)));
            input.addAll(sourceCommand);
            input.addObject().put("type", "canvas.edge.create").set("edge", mapper.valueToTree(Map.of(
                    "id", "$input", "kind", "execution", "sourceNodeId", "$source", "sourcePort", sourcePort,
                    "targetNodeId", "$agent", "targetPort", "inputs", "inputIndex", 0, "version", 0)));
        }
        String nodeId = mapper.valueToTree(mutations.apply(input, invocation(f, "agent-node-create"))).path("idMappings").path("$agent").asText();
        var plan = canvasRuns.plan(f.canvas(), new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(),
                CanvasRunData.Scope.DEPENDENCIES, List.of(nodeId), Map.of(), new CanvasRunData.Budget(4, 0, 0, 1, 1), new GenerationClientScope(f.user(), null)), actor(f));
        assertThat(plan.issues()).isEmpty();
        assertThat(plan.cost().agentRuns()).isEqualTo(1);
        var run = canvasRuns.start(f.canvas(), new CanvasRunData.Start("sdk-node", plan.id(), plan.hash(), new GenerationClientScope(f.user(), null)),
                ResourceCommandContext.user(actor(f), f.canvas()));
        await().atMost(Duration.ofSeconds(25)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
            var current = canvasRuns.view(run.id(), actor(f));
            assertThat(current.status()).withFailMessage("运行结果：%s", current).isEqualTo(CanvasRunData.State.SUCCEEDED);
            assertThat(current.nodes()).filteredOn(value -> value.nodeId().equals(nodeId)).singleElement().satisfies(value -> {
                assertThat(value.externalKind()).isEqualTo("AGENT");
                assertThat(value.outputs().at("results")).singleElement().satisfies(output -> assertThat(output.text()).contains("雨夜古城"));
            });
        });
        int expectedCalls = hasTool ? 2 : 1;
        assertThat(calls).hasValue(expectedCalls);
        var agentNode = canvasRuns.view(run.id(), actor(f)).nodes().stream().filter(value -> value.nodeId().equals(nodeId)).findFirst().orElseThrow();
        var job = nodeJobs.find(agentNode.id(), false);
        assertThat(job.scope().agentType()).isEqualTo(profile);
        if (hasResource) assertThat(job.scope().resources()).containsKey(CanvasAgentNodeScope.key(CanvasValueType.fromCode(inputType), f.project()));
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE agent_run_id=?", Long.class, job.agentRunId())).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_agent_model_request m JOIN afv_agent_run r ON r.run_id=m.run_id WHERE JSON_UNQUOTE(JSON_EXTRACT(r.canvas_scope_json,'$.nodeRunId'))=? AND m.status='COMPLETED' AND m.permit_held=FALSE", Long.class,
                agentNode.id())).isEqualTo((long) expectedCalls);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_agent_node_job WHERE canvas_run_id=?", Long.class, run.id())).isEqualTo(1L);
    }

    private void professionalInputTree(Fixture f) {
        long id = f.project();
        jdbc.update("INSERT INTO afv_script_episode(id,script_id,episode_number,title) VALUES(?,?,1,'雨夜古城')", id, id);
        jdbc.update("INSERT INTO afv_script_scene_item(id,script_id,episode_id,scene_number,scene_heading) VALUES(?,?,?,'1','古城夜景')", id, id, id);
        jdbc.update("INSERT INTO afv_storyboard(id,project_id,script_id,title) VALUES(?,?,?,'雨夜分镜')", id, id, id);
        jdbc.update("INSERT INTO afv_storyboard_episode(id,storyboard_id,script_episode_id,episode_number,title) VALUES(?,?,?,1,'第一集')", id, id, id);
        jdbc.update("INSERT INTO afv_storyboard_scene(id,storyboard_id,episode_id) VALUES(?,?,?)", id, id, id);
        jdbc.update("INSERT INTO afv_storyboard_item(id,storyboard_id,storyboard_episode_id,storyboard_scene_id) VALUES(?,?,?,?)", id, id, id, id);
        jdbc.update("INSERT INTO afv_asset(id,project_id,name,type,owner_type,owner_id,user_id) VALUES(?,?,'阿青','character',1,?,?)", id, id, f.user(), f.user());
        jdbc.update("INSERT INTO afv_asset_item(id,asset_id,name,item_type) VALUES(?,?,'雨夜造型','original')", id, id);
    }

    @Test void professionalScopeAdvancesOwnVersionsButRejectsFreshVersionOverwriteAfterHumanEdit() throws Exception {
        var f = professional(fixture());
        var created = resources.execute(new ResourceCreateCommand("scope-create", CanvasValueType.SCRIPT_EPISODE, f.project(),
                mapper.valueToTree(Map.of("scriptId", f.project(), "parentVersion", 0, "episodeNumber", 1, "title", "第一集"))), invocation(f, "create").requireMutationContext());
        long episode = created.result().path("id").asLong();
        var changed = resources.patch(new ResourcePatchCommand("scope-patch", CanvasValueType.SCRIPT_EPISODE, episode, 0,
                mapper.valueToTree(Map.of("title", "智能体修改"))), invocation(f, "patch").requireMutationContext());
        assertThat(changed.result().path("version").asInt()).isEqualTo(1);
        var scope = nodeJobs.find(f.scope().nodeRunId(), false).scope();
        assertThat(scope.resources().get(CanvasAgentNodeScope.key(CanvasValueType.SCRIPT, f.project())).version()).isEqualTo(2);
        assertThat(scope.resources().get(CanvasAgentNodeScope.key(CanvasValueType.SCRIPT_EPISODE, episode)).changed()).isTrue();
        resources.patch(new ResourcePatchCommand("human-edit", CanvasValueType.SCRIPT_EPISODE, episode, 1,
                mapper.valueToTree(Map.of("title", "用户的新版本"))), ResourceCommandContext.user(actor(f), f.canvas()));
        assertThatThrownBy(() -> resources.patch(new ResourcePatchCommand("scope-stale-input", CanvasValueType.SCRIPT_EPISODE, episode, 2,
                mapper.valueToTree(Map.of("title", "试图读取新版本后覆盖"))), invocation(f, "stale").requireMutationContext()))
                .hasMessageContaining("专业智能体执行期间业务内容已被其他操作修改");
        assertThat(jdbc.queryForObject("SELECT title FROM afv_script_episode WHERE id=?", String.class, episode)).isEqualTo("用户的新版本");
        assertThat(jdbc.queryForObject("SELECT version FROM afv_script WHERE id=?", Integer.class, f.project())).isEqualTo(3);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE request_id='scope-stale-input'", Long.class)).isZero();
    }

    @Test void agentAndTextShareModelCapacityAndLateResponseReleasesOnlyItsOriginalPermit() throws Exception {
        var f = fixture();
        var model = model(f);
        var call = modelRequests.admit(UUID.randomUUID().toString(), model, runtime(f), AgentKernelKey.modelRecordFingerprint(model)).orElseThrow();
        var second = fixture();
        assertThat(modelRequests.admit(UUID.randomUUID().toString(), model, runtime(second), AgentKernelKey.modelRecordFingerprint(model))).isEmpty();
        String sourceNode = mapper.valueToTree(mutations.apply(nodes("$quota", "文本并发验收"), invocation(f, "quota-source")))
                .path("idMappings").path("$quota").asText();
        var plan = canvasRuns.plan(f.canvas(), new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(),
                CanvasRunData.Scope.SINGLE, List.of(sourceNode), Map.of(), new CanvasRunData.Budget(2, 0, 0, 1, 1), new GenerationClientScope(f.user(), null)), actor(f));
        var canvasRun = canvasRuns.start(f.canvas(), new CanvasRunData.Start("quota-source-run", plan.id(), plan.hash(), new GenerationClientScope(f.user(), null)),
                ResourceCommandContext.user(actor(f), f.canvas()));
        String textId = UUID.randomUUID().toString();
        jdbc.update("""
                INSERT INTO afv_canvas_text_job(id,node_run_id,run_id,project_id,model_id,model_limit,input_json,deadline_at,next_run_at,created_at,updated_at)
                VALUES(?,?,?,?,?,1,'{}',DATE_ADD(UTC_TIMESTAMP(6),INTERVAL 1 HOUR),UTC_TIMESTAMP(6),UTC_TIMESTAMP(6),UTC_TIMESTAMP(6))
                """, textId, canvasRun.nodes().getFirst().id(), canvasRun.id(), f.project(), model.getId());
        var textClaim = transactions.execute(status -> textJobs.claim("quota-test"));
        assertThat(textClaim).isEmpty();
        modelRequests.uncertain(call, true);
        assertThat(modelRequests.admit(UUID.randomUUID().toString(), model, runtime(second), AgentKernelKey.modelRecordFingerprint(model))).isEmpty();
        assertThatThrownBy(() -> modelRequests.admit(UUID.randomUUID().toString(), model, runtime(f), AgentKernelKey.modelRecordFingerprint(model))).hasMessageContaining("原模型调用仍需核实");
        jdbc.update("UPDATE afv_agent_run SET owner_epoch=owner_epoch+1 WHERE run_id=?", f.runId());
        assertThatThrownBy(() -> modelRequests.admit(UUID.randomUUID().toString(), model, runtime(f), AgentKernelKey.modelRecordFingerprint(model))).hasMessageContaining("身份已失效");
        jdbc.update("UPDATE afv_canvas_text_job SET status='UNKNOWN_SUBMISSION',permit_held=TRUE WHERE id=?", textId);
        modelRequests.completed(call, ChatUsage.builder().inputTokens(11).outputTokens(7).build());
        assertThat(jdbc.queryForObject("SELECT input_tokens FROM afv_agent_model_call_usage WHERE model_call_id=?", Long.class, call.id())).isEqualTo(11L);
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_agent_model_request WHERE id=?", Boolean.class, call.id())).isFalse();
        assertThat(modelRequests.admit(UUID.randomUUID().toString(), model, runtime(second), AgentKernelKey.modelRecordFingerprint(model))).isEmpty();
        jdbc.update("UPDATE afv_canvas_text_job SET status='CANCELLED',permit_held=FALSE WHERE id=?", textId);
        var next = modelRequests.admit(UUID.randomUUID().toString(), model, runtime(second), AgentKernelKey.modelRecordFingerprint(model)).orElseThrow();
        modelRequests.completed(next, null);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 3})
    void professionalChildWaitReleasesParentAndResumesFromThePersistedChildResult(int childCount) throws Exception {
        var f = fixture();
        var model = model(f);
        var created = resources.execute(new ResourceCreateCommand("child-fixture", CanvasValueType.SCRIPT_EPISODE, f.project(),
                mapper.valueToTree(Map.of("scriptId", f.project(), "parentVersion", 0, "episodeNumber", 1, "title", "雨夜"))),
                ResourceCommandContext.user(actor(f), f.canvas()));
        long episode = created.result().path("id").asLong();
        for (int index = 0; index < 50 && resourceEvents.publishNext(); index++) { }
        var parentCalls = new AtomicInteger();
        var childCalls = new AtomicInteger();
        var lostInitialDispatch = new AtomicBoolean();
        doAnswer(invocation -> {
            StartAgentExecutionCommand command = invocation.getArgument(0);
            if (command.runtimeContextRequest().parentRun() != null
                    && command.runtimeContextRequest().project().projectId().equals(f.project())
                    && lostInitialDispatch.compareAndSet(false, true)) {
                jdbc.update("UPDATE afv_agent_run SET owner_instance_id='departed-child-owner',lease_until=DATE_SUB(UTC_TIMESTAMP(3),INTERVAL 1 SECOND) WHERE run_id=?", command.run().runId());
                return Mono.empty();
            }
            return invocation.callRealMethod();
        }).when(supervisor).start(any(StartAgentExecutionCommand.class));
        when(providers.agentScopeModelFingerprint(any(AiModel.class))).thenAnswer(call -> AgentKernelKey.modelRecordFingerprint(call.getArgument(0)));
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                new ChatModelBase() {
                    @Override public String getModelName() { return "test-child-wait"; }
                    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
                        if (schema.stream().anyMatch(tool -> tool.getName().equals("episode_script_creator"))) {
                            if (parentCalls.getAndIncrement() == 0) {
                                List<ContentBlock> calls = new ArrayList<>();
                                for (int index = 0; index < childCount; index++) calls.add(ToolUseBlock.builder().id("create-episode-child-" + index)
                                        .name("episode_script_creator").input(Map.of("scriptEpisodeId", episode))
                                        .content("{\"scriptEpisodeId\":" + episode + "}").build());
                                return Flux.just(ChatResponse.builder().id("parent-delegates").content(calls).finishReason("tool_calls").build());
                            }
                            assertThat(new CanvasJson(mapper).write(messages)).contains("分集创作结果已确认");
                            var converter = new OpenAIMessageConverter(message -> "", blocks -> blocks.toString());
                            var returnedIds = messages.stream().filter(message -> message.hasContentBlocks(ToolResultBlock.class))
                                    .map(message -> converter.convertToMessage(message, false).getToolCallId()).toList();
                            assertThat(returnedIds).hasSize(childCount);
                            for (int index = 0; index < childCount; index++) assertThat(returnedIds).contains("create-episode-child-" + index);
                            return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("已收到真实子任务结果").build()))
                                    .finishReason("stop").build());
                        }
                        childCalls.incrementAndGet();
                        return Mono.delay(Duration.ofMillis(750)).map(ignored -> ChatResponse.builder().content(List.of(TextBlock.builder()
                                .text("{\"scriptEpisodeId\":" + episode + ",\"sceneCount\":0,\"status\":\"completed\",\"message\":\"分集创作结果已确认\"}").build()))
                                .finishReason("stop").build()).flux();
                    }
                }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        var input = nodes("$parent", "父子任务验收");
        ((ObjectNode) input.get(0).path("node")).put("kind", "agent").set("config", mapper.valueToTree(Map.of(
                "agentType", "story_to_script", "modelId", model.getId(), "inputType", "text", "resultType", "text",
                "instruction", "检查本集内容并委派子任务", "maxAgentRuns", childCount + 1)));
        String nodeId = mapper.valueToTree(mutations.apply(input, invocation(f, "child-node"))).path("idMappings").path("$parent").asText();
        var plan = canvasRuns.plan(f.canvas(), new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(),
                CanvasRunData.Scope.SINGLE, List.of(nodeId), Map.of(), new CanvasRunData.Budget(childCount + 2, 0, 0, 1, 1), new GenerationClientScope(f.user(), null)), actor(f));
        assertThat(plan.issues()).isEmpty();
        var run = canvasRuns.start(f.canvas(), new CanvasRunData.Start("child-sdk", plan.id(), plan.hash(), new GenerationClientScope(f.user(), null)),
                ResourceCommandContext.user(actor(f), f.canvas()));
        String nodeRunId = run.nodes().getFirst().id();
        await().atMost(Duration.ofSeconds(30)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
            var current = canvasRuns.view(run.id(), actor(f));
            assertThat(current.status()).withFailMessage("父子任务结果：%s", current).isEqualTo(CanvasRunData.State.SUCCEEDED);
        });
        var node = canvasRuns.view(run.id(), actor(f)).nodes().getFirst();
        assertThat(node.outputs().at("results")).singleElement().satisfies(result -> assertThat(result.text()).contains("真实子任务结果"));
        assertThat(parentCalls).hasValue(2);
        assertThat(childCalls).hasValue(childCount);
        assertAutomaticNodeTasks(nodeRunId);
        assertThat(lostInitialDispatch).isTrue();
        assertThat(jdbc.queryForObject("SELECT MAX(owner_epoch) FROM afv_agent_run WHERE JSON_UNQUOTE(JSON_EXTRACT(canvas_scope_json,'$.nodeRunId'))=? AND parent_run_id IS NOT NULL", Long.class, node.id())).isGreaterThan(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_agent_run WHERE JSON_UNQUOTE(JSON_EXTRACT(canvas_scope_json,'$.nodeRunId'))=? AND status='COMPLETED'", Long.class, node.id())).isEqualTo(childCount + 1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_agent_event WHERE run_id=? AND raw_event_type='PLATFORM_REQUIRE_EXTERNAL_EXECUTION'", Long.class, node.externalId())).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT content FROM afv_agent_message WHERE run_id=? AND role='tool' AND tool_call_id='create-episode-child-0' AND tool_status='success'", String.class,
                node.externalId())).contains("分集创作结果已确认");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_agent_model_request m JOIN afv_agent_run r ON r.run_id=m.run_id WHERE JSON_UNQUOTE(JSON_EXTRACT(r.canvas_scope_json,'$.nodeRunId'))=? AND m.permit_held=TRUE", Long.class, node.id())).isZero();
    }

    @Test void professionalScriptNodeAutomaticallyPublishesTheActuallyCommittedBusinessResult() throws Exception {
        var f = fixture();
        var model = model(f);
        var calls = new AtomicInteger();
        var arguments = Map.<String, Object>of("scriptId", f.project(), "expectedVersion", 0, "episodeNumber", 1,
                "title", "雨夜来信", "synopsis", "阿青在雨夜收到一封信", "rawContent", "第一集：雨夜来信", "sourceType", 2);
        String encoded = mapper.writeValueAsString(arguments);
        when(providers.agentScopeModelFingerprint(any(AiModel.class))).thenAnswer(call -> AgentKernelKey.modelRecordFingerprint(call.getArgument(0)));
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                new ChatModelBase() {
                    @Override public String getModelName() { return "professional-script-contract"; }
                    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
                        assertThat(schema).anyMatch(tool -> tool.getName().equals("save_script_episode"));
                        if (calls.getAndIncrement() == 0) return Flux.just(ChatResponse.builder().id("save-episode")
                                .content(List.of(ToolUseBlock.builder().id("save-real-episode").name("save_script_episode")
                                        .input(arguments).content(encoded).build())).finishReason("tool_calls").build());
                        assertThat(new CanvasJson(mapper).write(messages)).contains("episode_version", "success");
                        return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("分集已写入正式剧本").build()))
                                .finishReason("stop").build());
                    }
                }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        var input = nodes("$script", "原剧本");
        var source = (ObjectNode) input.get(0).path("node");
        source.put("kind", "script").set("config", mapper.createObjectNode());
        source.set("resource", mapper.valueToTree(Map.of("type", "script", "id", f.project(), "version", 0)));
        var operation = nodes("$parse", "解析剧本");
        ((ObjectNode) operation.get(0).path("node")).put("kind", "scriptParse").set("config", mapper.valueToTree(Map.of(
                "agentType", "script_full_parse", "modelId", model.getId(), "instruction", "创建雨夜来信分集", "maxAgentRuns", 1)));
        input.addAll(operation);
        input.addObject().put("type", "canvas.edge.create").set("edge", mapper.valueToTree(Map.of("id", "$input", "kind", "execution",
                "sourceNodeId", "$script", "sourcePort", "script", "targetNodeId", "$parse", "targetPort", "script", "inputIndex", 0, "version", 0)));
        var ids = mapper.valueToTree(mutations.apply(input, invocation(f, "script-plan"))).path("idMappings");
        var plan = canvasRuns.plan(f.canvas(), new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(),
                CanvasRunData.Scope.DEPENDENCIES, List.of(ids.path("$parse").asText()), Map.of(),
                new CanvasRunData.Budget(4, 0, 0, 1, 1), new GenerationClientScope(f.user(), null)), actor(f));
        assertThat(plan.issues()).isEmpty();
        var run = canvasRuns.start(f.canvas(), new CanvasRunData.Start("professional-script", plan.id(), plan.hash(), new GenerationClientScope(f.user(), null)),
                ResourceCommandContext.user(actor(f), f.canvas()));
        String nodeRunId = run.nodes().stream().filter(node -> node.nodeId().equals(ids.path("$parse").asText())).findFirst().orElseThrow().id();
        await().atMost(Duration.ofSeconds(25)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
            var current = canvasRuns.view(run.id(), actor(f));
            assertThat(current.status()).withFailMessage("正式业务写入结果：%s", current).isEqualTo(CanvasRunData.State.SUCCEEDED);
            var node = current.nodes().stream().filter(item -> item.id().equals(nodeRunId)).findFirst().orElseThrow();
            assertThat(node.outputs().at("script")).singleElement().satisfies(output -> {
                assertThat(output.resource().id()).isEqualTo(f.project());
                assertThat(output.resource().version()).isEqualTo(1);
            });
        });
        assertThat(calls).hasValue(2);
        assertThat(jdbc.queryForObject("SELECT title FROM afv_script_episode WHERE script_id=? AND deleted=FALSE", String.class, f.project())).isEqualTo("雨夜来信");
        var task = agentDetails.detail(run.id(), nodeRunId, actor(f)).tasks().getFirst();
        assertAutomaticNodeTasks(nodeRunId);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE agent_run_id=? AND tool_call_id='save-real-episode' AND origin='agent'", Long.class, task.id())).isEqualTo(1);
    }

    @ParameterizedTest @ValueSource(booleans = {false, true})
    void professionalBatchRunsAutomaticallyAndAdvancesOnlyItsOwnBusinessVersions(boolean retryFirst) throws Exception {
        var f = fixture();
        var model = model(f);
        var calls = new AtomicInteger();
        when(providers.agentScopeModelFingerprint(any(AiModel.class))).thenAnswer(call -> AgentKernelKey.modelRecordFingerprint(call.getArgument(0)));
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                new ChatModelBase() {
                    @Override public String getModelName() { return "batch-script-contract"; }
                    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
                        int current = calls.getAndIncrement();
                        if (retryFirst && current == 0) return Flux.just(ChatResponse.builder()
                                .content(List.of(TextBlock.builder().text("本项未保存，需要重新执行").build())).finishReason("stop").build());
                        boolean finished = retryFirst ? current == 2 || current == 4 : current % 2 == 1;
                        if (finished) return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("本项分集已保存").build()))
                                .finishReason("stop").build());
                        int number = retryFirst ? current == 1 ? 2 : 1 : current / 2 + 1;
                        int version = retryFirst ? current == 1 ? 0 : 1 : number - 1;
                        var args = Map.<String, Object>of("scriptId", f.project(), "expectedVersion", version,
                                "episodeNumber", number, "title", "批量第" + number + "集");
                        return Flux.just(ChatResponse.builder().id("batch-save-" + number)
                                .content(List.of(ToolUseBlock.builder().id("batch-episode-" + number).name("save_script_episode")
                                        .input(args).content(new CanvasJson(mapper).write(args)).build())).finishReason("tool_calls").build());
                    }
                }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        var input = nodes("$one", "第一项故事");
        input.addAll(nodes("$two", "第二项故事"));
        var batch = nodes("$batch", "逐项创作剧本");
        ((ObjectNode) batch.get(0).path("node")).put("kind", "foreach").set("config", mapper.valueToTree(Map.of(
                "inputType", "text", "resultType", "script", "maxItems", 2, "operationKind", "storyToScript", "inputPort", "story", "outputPort", "script",
                "operationConfig", Map.of("agentType", "story_to_script", "modelId", model.getId(), "maxAgentRuns", 1))));
        input.addAll(batch);
        int index = 0;
        for (String source : List.of("$one", "$two")) input.addObject().put("type", "canvas.edge.create").set("edge", mapper.valueToTree(Map.of(
                "id", "$edge" + index, "kind", "execution", "sourceNodeId", source, "sourcePort", "text",
                "targetNodeId", "$batch", "targetPort", "items", "inputIndex", index++, "version", 0)));
        var ids = mapper.valueToTree(mutations.apply(input, invocation(f, "batch-create"))).path("idMappings");
        var plan = canvasRuns.plan(f.canvas(), new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(),
                CanvasRunData.Scope.DEPENDENCIES, List.of(ids.path("$batch").asText()), Map.of(), new CanvasRunData.Budget(retryFirst ? 6 : 5, 0, 0, 2, 2),
                new GenerationClientScope(f.user(), null)), actor(f));
        assertThat(plan.issues()).isEmpty();
        assertThat(plan.cost().nodes()).isEqualTo(5);
        assertThat(plan.cost().agentRuns()).isEqualTo(2);
        var run = canvasRuns.start(f.canvas(), new CanvasRunData.Start("professional-batch", plan.id(), plan.hash(), new GenerationClientScope(f.user(), null)),
                ResourceCommandContext.user(actor(f), f.canvas()));
        if (retryFirst) {
            await().atMost(Duration.ofSeconds(30)).pollInterval(Duration.ofMillis(200)).untilAsserted(() ->
                    assertThat(canvasRuns.view(run.id(), actor(f)).nodes()).anyMatch(node ->
                            "BATCH".equals(node.externalKind()) && node.status() == CanvasRunData.State.WAITING_CONFIRMATION));
            var parent = canvasRuns.view(run.id(), actor(f)).nodes().stream().filter(node -> "BATCH".equals(node.externalKind())).findFirst().orElseThrow();
            String review = mapper.writeValueAsString(Map.of("canvasRunId", run.id(), "nodeRunId", parent.id(), "expectedRevision", parent.revision(),
                    "action", "RETRY_FAILED", "note", "重试第一项未写入的分集"));
            for (int repeated = 0; repeated < 2; repeated++)
                assertThat(mapper.readTree(tools.findExecutor("review_canvas_batch").execute(review, invocation(f, "review-batch"))).path("status").asText()).isNotEqualTo("error");
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_node_run WHERE parent_node_run_id=? AND attempt=2", Long.class, parent.id())).isEqualTo(1);
        }
        await().atMost(Duration.ofSeconds(30)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
            var current = canvasRuns.view(run.id(), actor(f));
            assertThat(current.status()).withFailMessage("专业批量结果：%s", current).isEqualTo(CanvasRunData.State.SUCCEEDED);
            assertThat(current.nodes().stream().filter(node -> "BATCH".equals(node.externalKind())).findFirst().orElseThrow().outputs().at("results"))
                    .extracting(value -> value.resource().version()).containsExactlyElementsOf(retryFirst ? List.of(2L, 1L) : List.of(1L, 2L));
        });
        assertThat(calls).hasValue(retryFirst ? 5 : 4);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_script_episode WHERE script_id=? AND deleted=FALSE", Long.class, f.project())).isEqualTo(2L);
    }

    @Test void professionalPlanRejectsMissingTypedInputAndImpossibleOutputsBeforeAnyModelCall() throws Exception {
        var f = fixture();
        var model = model(f);
        when(providers.agentScopeModelFingerprint(any(AiModel.class))).thenAnswer(call -> AgentKernelKey.modelRecordFingerprint(call.getArgument(0)));
        var input = nodes("$missing", "缺少剧本输入");
        ((ObjectNode) input.get(0).path("node")).put("kind", "agent").set("config", mapper.valueToTree(Map.of(
                "agentType", "script_full_parse", "modelId", model.getId(), "inputType", "script", "resultType", "text", "instruction", "解析剧本", "maxAgentRuns", 1)));
        var invalid = nodes("$wrong", "不能生成图片的分析节点");
        ((ObjectNode) invalid.get(0).path("node")).put("kind", "agent").set("config", mapper.valueToTree(Map.of(
                "agentType", "concept_visualizer", "modelId", model.getId(), "inputType", "text", "resultType", "image", "instruction", "检查内容", "maxAgentRuns", 1)));
        input.addAll(invalid);
        mutations.apply(input, invocation(f, "invalid-professional-plan"));
        var plan = canvasRuns.plan(f.canvas(), new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(),
                CanvasRunData.Scope.ALL, List.of(), Map.of(), new CanvasRunData.Budget(5, 0, 0, 1, 1), new GenerationClientScope(f.user(), null)), actor(f));
        assertThat(plan.issues()).anyMatch(issue -> issue.code().equals("AGENT_INPUT_INVALID"));
        assertThat(plan.issues()).anyMatch(issue -> issue.message().contains("不能产生"));
        verify(providers, never()).provisionSingleRequestModel(any(AiModel.class));
    }

    @Test void aReceivedModelFailureReleasesItsOwnPermitAndAllowsTheNextRequest() throws Exception {
        var f = fixture();
        var model = model(f);
        var call = modelRequests.admit(UUID.randomUUID().toString(), model, runtime(f), AgentKernelKey.modelRecordFingerprint(model)).orElseThrow();
        modelRequests.uncertain(call, false);
        modelRequests.failed(call);
        modelRequests.failed(call);
        modelRequests.completed(call, null);
        assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_model_request WHERE id=?", String.class, call.id())).isEqualTo("COMPLETED");
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_agent_model_request WHERE id=?", Boolean.class, call.id())).isFalse();
        assertThat(jdbc.queryForObject("SELECT response_received_at IS NOT NULL FROM afv_agent_model_request WHERE id=?", Boolean.class, call.id())).isTrue();
        assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_model_call_usage WHERE model_call_id=?", String.class, call.id())).isEqualTo("FAILED");
        assertThat(modelRequests.admit(UUID.randomUUID().toString(), model, runtime(f), AgentKernelKey.modelRecordFingerprint(model))).isPresent();
    }

    @Test void uncertainModelCallRequiresAuditedAdminConfirmationAndDoesNotReplayTheProvider() throws Exception {
        var f = fixture();
        var model = model(f);
        var call = modelRequests.admit(UUID.randomUUID().toString(), model, runtime(f), AgentKernelKey.modelRecordFingerprint(model)).orElseThrow();
        modelRequests.uncertain(call, false);
        assertThatThrownBy(() -> recovery.get(ExecutionRecoveryData.Kind.AGENT_MODEL, call.id(), f.user())).hasMessageContaining("管理员");
        Long role = jdbc.queryForObject("SELECT id FROM sys_role WHERE UPPER(code)='ADMIN' AND deleted=FALSE", Long.class);
        jdbc.update("INSERT INTO sys_user_role(user_id,role_id) VALUES(?,?)", f.user(), role);
        var pending = recovery.get(ExecutionRecoveryData.Kind.AGENT_MODEL, call.id(), f.user());
        assertThat(pending.canResolve()).isTrue();
        var request = new ExecutionRecoveryData.Decision(f.user(), "review-" + call.id(), pending.revision(), pending.factsHash(),
                "测试供应商已确认原调用终止", true);
        var receipt = recovery.confirm(ExecutionRecoveryData.Kind.AGENT_MODEL, call.id(), request, f.user());
        assertThat(receipt.status()).isEqualTo("CANCELLED");
        assertThat(recovery.confirm(ExecutionRecoveryData.Kind.AGENT_MODEL, call.id(), request, f.user())).isEqualTo(receipt);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_execution_recovery_decision WHERE job_id=?", Long.class, call.id())).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT permit_held FROM afv_agent_model_request WHERE id=?", Boolean.class, call.id())).isFalse();
    }

    @Test void currentModelAndApiChangesStopNewRequestsBeforeReservingQuota() throws Exception {
        var f = fixture();
        var model = model(f);
        String original = AgentKernelKey.modelRecordFingerprint(model);
        jdbc.update("UPDATE afv_ai_model SET code='changed-model' WHERE id=?", model.getId());
        assertThatThrownBy(() -> modelRequests.admit(UUID.randomUUID().toString(), model, runtime(f), original)).hasMessageContaining("配置在执行期间变化");
        jdbc.update("UPDATE afv_ai_model SET code=? WHERE id=?", model.getCode(), model.getId());
        when(providers.agentScopeModelFingerprint(any(AiModel.class))).thenReturn("changed-api-endpoint");
        assertThatThrownBy(() -> modelRequests.admit(UUID.randomUUID().toString(), model, runtime(f), original)).hasMessageContaining("配置在执行期间变化");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_agent_model_request WHERE run_id=?", Long.class, f.runId())).isZero();
        verify(providers, never()).provisionSingleRequestModel(any(AiModel.class));
    }

    @Test void explicitReasoningLevelIsPreservedWhenCheckingCurrentProviderConfiguration() throws Exception {
        var f = fixture();
        var model = model(f);
        jdbc.update("UPDATE afv_ai_model SET support_reasoning=TRUE,reasoning_effort_levels=JSON_ARRAY('low','high') WHERE id=?", model.getId());
        model = AiModelRequestOptions.withReasoningEffort(modelService.getById(model.getId()), "high", mapper);
        var claim = modelRequests.admit(UUID.randomUUID().toString(), model, runtime(f), AgentKernelKey.modelRecordFingerprint(model)).orElseThrow();
        modelRequests.completed(claim, null);
        assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_model_request WHERE id=?", String.class, claim.id())).isEqualTo("COMPLETED");
    }

    @Test void historicalGenerationLibraryProvidesVersionedSourcesWithoutOtherProjectContent() throws Exception {
        var f = fixture();
        long otherProject = IDS.incrementAndGet();
        jdbc.update("INSERT INTO afv_project(id,name,owner_type,owner_id) VALUES(?,'其他项目',1,?)", otherProject, f.user());
        for (long project : List.of(f.project(), otherProject)) {
            jdbc.update("INSERT INTO afv_image_task(id,task_id,user_id,project_id,prompt,status,owner_type,owner_id) VALUES(?,?,?,?,?,2,1,?)",
                    project, "history-" + project, f.user(), project, project == f.project() ? "当前画布的历史图片" : "其他项目的秘密提示词", f.user());
            jdbc.update("INSERT INTO afv_image_item(id,task_id,image_url,status,version) VALUES(?,?,?,1,3)", project, project, "https://example.invalid/historical.png");
        }
        var directory = mapper.readTree(tools.findExecutor("query_canvas_library").execute("{\"category\":\"imageHistory\"}", invocation(f, "image-history")));
        assertThat(directory.path("status").asText()).isEqualTo("success");
        assertThat(directory.path("items")).hasSize(1);
        assertThat(directory.toString()).doesNotContain("秘密提示词", "historical.png");
        var items = mapper.readTree(tools.findExecutor("query_canvas_library").execute("{\"category\":\"imageHistory\",\"parentId\":" + f.project() + "}", invocation(f, "image-results")));
        var source = items.path("items").get(0).path("sources").get(0);
        assertThat(source.path("id").asLong()).isEqualTo(f.project());
        assertThat(source.path("version").asInt()).isEqualTo(3);
        assertThat(source.path("type").asText()).isEqualTo("imageItem");
        assertThat(source.path("field").asText()).isEqualTo("imageUrl");
        var denied = mapper.readTree(tools.findExecutor("query_canvas_library").execute("{\"category\":\"imageHistory\",\"parentId\":" + otherProject + "}", invocation(f, "other-history")));
        assertThat(denied.path("code").asInt()).isEqualTo(403);
    }

    @Test void professionalMediaInputIsDeliveredAsRealPrivateImageBytesToTheSelectedModel() throws Exception {
        var f = fixture();
        var model = model(f);
        jdbc.update("UPDATE afv_ai_model SET multimodal_input_types=JSON_ARRAY('image'),multimodal_input_transports=JSON_OBJECT('image',JSON_ARRAY('base64')) WHERE id=?", model.getId());
        if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config WHERE is_default=TRUE AND deleted=FALSE", Long.class) == 0)
            jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('智能体测试存储','local',?,TRUE)", FILES.toString());
        var bytes = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB), "png", bytes);
        byte[] png = bytes.toByteArray();
        var media = intake.upload(new ByteArrayInputStream(png), "private-input.png", "image/png", UUID.randomUUID().toString(), f.project(), ResourceCommandContext.user(actor(f), f.canvas()));
        var seen = new AtomicInteger();
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                new ChatModelBase() {
                    @Override public String getModelName() { return "private-image-observer"; }
                    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
                        var observed = messages.stream().flatMap(message -> message.getContent().stream()).filter(ImageBlock.class::isInstance).map(ImageBlock.class::cast).toList();
                        assertThat(observed).singleElement().satisfies(block -> {
                            assertThat(block.getSource()).isInstanceOf(Base64Source.class);
                            assertThat(Base64.getDecoder().decode(((Base64Source) block.getSource()).getData())).isEqualTo(png);
                        });
                        seen.incrementAndGet();
                        return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("已读取所提供的图片").build())).finishReason("stop").build());
                    }
                }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        var input = nodes("$image", "私有参考图");
        ((ObjectNode) input.get(0).path("node")).put("kind", "image").set("config", mapper.valueToTree(Map.of("resourceId", media.id())));
        var agent = nodes("$observe", "观察参考图");
        ((ObjectNode) agent.get(0).path("node")).put("kind", "agent").set("config", mapper.valueToTree(Map.of(
                "agentType", "concept_visualizer", "modelId", model.getId(), "inputType", "image", "resultType", "text", "instruction", "描述参考图", "maxAgentRuns", 1)));
        input.addAll(agent);
        input.addObject().put("type", "canvas.edge.create").set("edge", mapper.valueToTree(Map.of("id", "$image-edge", "kind", "execution",
                "sourceNodeId", "$image", "sourcePort", "media", "targetNodeId", "$observe", "targetPort", "inputs", "inputIndex", 0, "version", 0)));
        var ids = mapper.valueToTree(mutations.apply(input, invocation(f, "media-agent-create"))).path("idMappings");
        var plan = canvasRuns.plan(f.canvas(), new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(), CanvasRunData.Scope.DEPENDENCIES,
                List.of(ids.path("$observe").asText()), Map.of(), new CanvasRunData.Budget(3, 0, 0, 1, 1), new GenerationClientScope(f.user(), null)), actor(f));
        assertThat(plan.issues()).isEmpty();
        var run = canvasRuns.start(f.canvas(), new CanvasRunData.Start("private-media-sdk", plan.id(), plan.hash(), new GenerationClientScope(f.user(), null)),
                ResourceCommandContext.user(actor(f), f.canvas()));
        await().atMost(Duration.ofSeconds(25)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
            var current = canvasRuns.view(run.id(), actor(f));
            assertThat(current.status()).withFailMessage("私有媒体运行结果：%s", current).isEqualTo(CanvasRunData.State.SUCCEEDED);
        });
        assertThat(seen).hasValue(1);
        String stored = jdbc.queryForObject("SELECT input_json FROM afv_canvas_agent_node_job WHERE canvas_run_id=?", String.class, run.id());
        assertThat(stored).contains(media.id()).doesNotContain(Base64.getEncoder().encodeToString(png));
    }

    @Test void mixedReferenceMediaReachesTheActualSdkOncePerFileAndUnsupportedModelsFailBeforeCalling() throws Exception {
        var f = fixture(); var model = model(f);
        if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config WHERE is_default=TRUE AND deleted=FALSE", Long.class) == 0)
            jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('智能体测试存储','local',?,TRUE)", FILES.toString());
        var files = CanvasAgentMediaFixtures.create(FILES.resolve("mixed-" + f.user()));
        var input = nodes("$observe", "混合媒体观察");
        ((ObjectNode) input.get(0).path("node")).put("kind", "agent").set("config", mapper.valueToTree(Map.of(
                "agentType", "concept_visualizer", "modelId", model.getId(), "inputType", "image", "resultType", "text",
                "instruction", "结合图片、视频、音频和文档说明素材内容，明确实际观察范围", "maxAgentRuns", 1)));
        var references = new ArrayList<String>();
        for (var file : files) {
            String suffix = switch (file.kind()) { case "image" -> ".png"; case "video" -> ".mp4"; case "audio" -> ".wav"; default -> ".pdf"; };
            var media = intake.upload(new ByteArrayInputStream(file.bytes()), "reference-" + file.kind() + suffix, file.mime(),
                    UUID.randomUUID().toString(), f.project(), ResourceCommandContext.user(actor(f), f.canvas()));
            references.add(media.id());
            var source = nodes("$" + file.kind(), "参考" + file.kind());
            ((ObjectNode) source.get(0).path("node")).put("kind", file.kind()).set("config", mapper.valueToTree(Map.of("resourceId", media.id())));
            input.addAll(source);
            input.addObject().put("type", "canvas.edge.create").set("edge", mapper.valueToTree(Map.of("id", "$ref-" + file.kind(), "kind", "execution",
                    "sourceNodeId", "$" + file.kind(), "sourcePort", "media", "targetNodeId", "$observe", "targetPort", file.port(), "inputIndex", 0, "version", 0)));
        }
        input.addObject().put("type", "canvas.edge.create").set("edge", mapper.valueToTree(Map.of("id", "$same-image", "kind", "execution",
                "sourceNodeId", "$image", "sourcePort", "media", "targetNodeId", "$observe", "targetPort", "inputs", "inputIndex", 0, "version", 0)));
        var ids = mapper.valueToTree(mutations.apply(input, invocation(f, "mixed-media-create"))).path("idMappings");
        var request = new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(), CanvasRunData.Scope.DEPENDENCIES,
                List.of(ids.path("$observe").asText()), Map.of(), new CanvasRunData.Budget(5, 0, 0, 1, 1), new GenerationClientScope(f.user(), null));
        assertThat(canvasRuns.plan(f.canvas(), request, actor(f)).issues()).anySatisfy(issue -> assertThat(issue.message()).contains("当前模型不支持"));
        verify(providers, never()).provisionSingleRequestModel(any(AiModel.class));
        jdbc.update("UPDATE afv_ai_model SET multimodal_input_types=JSON_ARRAY('image','video','audio','file'),multimodal_input_transports=JSON_OBJECT('image',JSON_ARRAY('base64'),'video',JSON_ARRAY('base64'),'audio',JSON_ARRAY('base64'),'file',JSON_ARRAY('base64')) WHERE id=?", model.getId());
        var seen = new AtomicInteger();
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                new ChatModelBase() {
                    @Override public String getModelName() { return "mixed-private-media-observer"; }
                    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
                        var blocks = messages.stream().flatMap(message -> message.getContent().stream()).toList();
                        for (var file : files) {
                            var observed = blocks.stream().map(block -> switch (block) {
                                case ImageBlock value when file.kind().equals("image") -> value.getSource();
                                case VideoBlock value when file.kind().equals("video") -> value.getSource();
                                case AudioBlock value when file.kind().equals("audio") -> value.getSource();
                                case DataBlock value when file.kind().equals("document") -> value.getSource();
                                default -> null;
                            }).filter(value -> value != null).toList();
                            assertThat(observed).singleElement().isInstanceOfSatisfying(Base64Source.class,
                                    source -> assertThat(Base64.getDecoder().decode(source.getData())).isEqualTo(file.bytes()));
                        }
                        assertThat(new CanvasJson(mapper).write(messages)).contains("视频采样和文档读取范围取决于模型能力");
                        seen.incrementAndGet();
                        return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("四类参考素材已接收，按实际提供内容分析").build())).finishReason("stop").build());
                    }
                }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        var plan = canvasRuns.plan(f.canvas(), request, actor(f));
        assertThat(plan.issues()).isEmpty();
        var run = canvasRuns.start(f.canvas(), new CanvasRunData.Start("mixed-media-sdk", plan.id(), plan.hash(), new GenerationClientScope(f.user(), null)),
                ResourceCommandContext.user(actor(f), f.canvas()));
        await().atMost(Duration.ofSeconds(30)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
            var current = canvasRuns.view(run.id(), actor(f));
            assertThat(current.status()).withFailMessage("混合媒体结果：%s", current).isEqualTo(CanvasRunData.State.SUCCEEDED);
        });
        assertThat(seen).hasValue(1);
        String stored = jdbc.queryForObject("SELECT input_json FROM afv_canvas_agent_node_job WHERE canvas_run_id=?", String.class, run.id());
        references.forEach(reference -> assertThat(stored).contains(reference));
        files.forEach(file -> assertThat(stored).doesNotContain(Base64.getEncoder().encodeToString(file.bytes())));
    }

    @ParameterizedTest @ValueSource(strings = {"DEFAULT", "ALWAYS_ALLOW"})
    void rootAssistantCreatesConnectsRunsWaitsAndWritesBackThroughActualSdkTools(String mode) throws Exception {
        var f = fixture();
        var model = model(f);
        var operations = nodes("$source", "助手文本");
        var combine = nodes("$combine", "整理文本");
        ((ObjectNode) combine.get(0).path("node")).put("kind", "textCombine").set("config", mapper.createObjectNode());
        operations.addAll(combine);
        var script = nodes("$script", "项目剧本");
        var target = (ObjectNode) script.get(0).path("node");
        target.put("kind", "script").set("config", mapper.createObjectNode());
        target.set("resource", mapper.valueToTree(Map.of("type", "script", "id", f.project(), "version", 0)));
        operations.addAll(script);
        operations.addObject().put("type", "canvas.edge.create").set("edge", mapper.valueToTree(Map.of("id", "$chain", "kind", "execution",
                "sourceNodeId", "$source", "sourcePort", "text", "targetNodeId", "$combine", "targetPort", "texts", "inputIndex", 0, "version", 0)));
        var steps = new AtomicInteger();
        var created = new AtomicReference<JsonNode>();
        var execution = new AtomicReference<String>();
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                new ChatModelBase() {
                    @Override public String getModelName() { return "root-canvas-scenario"; }
                    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
                        assertThat(schema).noneMatch(tool -> Set.of("run_shell", "execute_shell", "load_skill").contains(tool.getName()));
                        int step = steps.getAndIncrement();
                        String tool;
                        Map<String, Object> arguments;
                        switch (step) {
                            case 0 -> { tool = "get_canvas_context"; arguments = Map.of("kinds", List.of("text", "textCombine", "script")); }
                            case 1 -> { tool = "apply_canvas_operations"; arguments = Map.of("operations", mapper.convertValue(operations, List.class)); }
                            case 2 -> {
                                created.set(lastTool(messages, "apply_canvas_operations"));
                                assertThat(created.get().path("status").asText()).withFailMessage("图命令结果：%s", created.get()).isEqualTo("success");
                                tool = "validate_canvas_run"; arguments = Map.of("expectedRevision", created.get().path("canvasRevision").asLong(), "mode", "DEPENDENCIES",
                                        "nodeIds", List.of(created.get().path("idMappings").path("$combine").asText()), "reuse", Map.of(),
                                        "budget", Map.of("maxNodes", 3, "maxImages", 0, "maxVideoSeconds", 0, "maxParallel", 1, "maxBatchItems", 1));
                            }
                            case 3 -> {
                                var result = lastTool(messages, "validate_canvas_run");
                                assertThat(result.path("status").asText()).isEqualTo("ready");
                                tool = "run_canvas_nodes"; arguments = Map.of("planId", result.at("/plan/id").asText(), "planHash", result.at("/plan/hash").asText());
                            }
                            case 4 -> {
                                execution.set(lastTool(messages, "run_canvas_nodes").path("canvasRunId").asText());
                                tool = "await_canvas_run"; arguments = Map.of("canvasRunId", execution.get());
                            }
                            case 5 -> {
                                var result = lastTool(messages, "await_canvas_run");
                                assertThat(result.path("status").asText()).isEqualTo("SUCCEEDED");
                                String nodeRun = null;
                                for (var node : result.path("nodes")) if (node.path("nodeId").asText().equals(created.get().path("idMappings").path("$combine").asText())) nodeRun = node.path("nodeRunId").asText();
                                assertThat(nodeRun).isNotBlank();
                                tool = "query_canvas_outputs"; arguments = Map.of("canvasRunId", execution.get(), "nodeRunId", nodeRun, "port", "text");
                            }
                            case 6 -> {
                                var output = lastTool(messages, "query_canvas_outputs").path("items").get(0);
                                assertThat(output.path("excerpt").asText()).isEqualTo("正文");
                                tool = "bind_canvas_output"; arguments = Map.of("canvasRunId", execution.get(), "outputId", output.path("id").asText(),
                                        "targetNodeId", created.get().path("idMappings").path("$script").asText(), "binding", "script:summary", "expectedVersion", 0);
                            }
                            case 7 -> {
                                assertThat(lastTool(messages, "bind_canvas_output").path("status").asText()).isEqualTo("success");
                                tool = "focus_canvas_nodes"; arguments = Map.of("nodeIds", List.of(created.get().path("idMappings").path("$script").asText()));
                            }
                            default -> { return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("节点已创建、连接并完成运行，结果已写入剧本摘要").build())).finishReason("stop").build()); }
                        }
                        return Flux.just(ChatResponse.builder().content(List.of(ToolUseBlock.builder().id("root-step-" + step).name(tool)
                                .input(arguments).content(new CanvasJson(mapper).write(arguments)).build())).finishReason("tool_calls").build());
                    }
                }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        var started = pipeline.start(new AiChatReqVO().setModelId(model.getId()).setAgentType("canvas_assistant").setProjectId(f.project())
                .setCanvasContext(new CanvasAgentContext(f.canvas(), List.of(), List.of(), null, 1L)).setToolExecutionMode(mode)
                .setMessage("创建文本处理链路，运行后把结果保存为剧本摘要"), f.user()).block(Duration.ofSeconds(15));
        for (int step : mode.equals("DEFAULT") ? List.of(1, 3, 6) : List.of(3)) {
            await().atMost(Duration.ofSeconds(30)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
                assertThat(steps.get()).isGreaterThan(step);
                assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId()))
                        .withFailMessage("根助手状态：%s", jdbc.queryForMap("SELECT status,error_message FROM afv_agent_run WHERE run_id=?", started.runId()))
                        .isEqualTo("WAITING_CONFIRMATION");
            });
            var approval = new ToolConfirmationReqVO(); approval.setRunId(started.runId());
            approval.setReplyId(jdbc.queryForObject("SELECT waiting_reply_id FROM afv_agent_run WHERE run_id=?", String.class, started.runId()));
            var pending = waitingStates.getPendingConfirmationAuthorized(started.runId(), f.user(), approval.getReplyId()).block(Duration.ofSeconds(10));
            assertThat(pending.decisionIds()).containsExactly("root-step-" + step);
            var review = mapper.readTree(pending.pendingToolCallsJson()).get(0).path("review");
            if (step == 1) {
                assertThat(review.path("changes").toString()).contains("文本", "正文", "助手文本 → 整理文本").doesNotContain("canvas.node.create");
            } else if (step == 6) {
                assertThat(review.path("changes").toString()).contains("剧本摘要");
                assertThat(review.path("rows").toString()).contains("写入前", "（空）", "写入后", "正文");
            }
            var decision = new ToolConfirmationReqVO.DecisionVO(); decision.setToolCallId("root-step-" + step);
            decision.setApproved(true); approval.setDecisions(List.of(decision));
            confirmations.respond(approval, f.user()).block(Duration.ofSeconds(10));
        }
        await().atMost(Duration.ofSeconds(30)).pollInterval(Duration.ofMillis(200)).untilAsserted(() ->
                assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId())).isEqualTo("COMPLETED"));
        assertThat(canvases.snapshot(f.canvas(), actor(f)).graph().nodes()).hasSize(3);
        assertThat(canvases.snapshot(f.canvas(), actor(f)).graph().edges()).hasSize(1);
        assertThat(jdbc.queryForObject("SELECT summary FROM afv_script WHERE id=?", String.class, f.project())).isEqualTo("正文");
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE agent_run_id=? AND origin='agent'", Long.class, started.runId())).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_operation WHERE agent_run_id=?", Long.class, started.runId())).isEqualTo(1);
        assertThat(steps).hasValue(9);
    }
    @Test void assistantOrganizesOnlyTheRequestedExistingEpisodeAndMarksMissingFramesWithoutGeneration() throws Exception {
        var f = fixture();
        professionalInputTree(f);
        var model = model(f);
        long episode = IDS.incrementAndGet(), scene = IDS.incrementAndGet(), boardEpisode = IDS.incrementAndGet();
        long boardScene = IDS.incrementAndGet(), missing = IDS.incrementAndGet(), ready = IDS.incrementAndGet();
        jdbc.update("INSERT INTO afv_script_episode(id,script_id,episode_number,title) VALUES(?,?,2,'第二集 · 雨夜灯塔')", episode, f.project());
        jdbc.update("INSERT INTO afv_script_scene_item(id,script_id,episode_id,scene_number,scene_heading,character_asset_ids) VALUES(?,?,?,'2-1','灯塔外景',JSON_ARRAY(?))",
                scene, f.project(), episode, f.project());
        jdbc.update("INSERT INTO afv_storyboard_episode(id,storyboard_id,script_episode_id,episode_number,title) VALUES(?,?,?,2,'第二集 · 雨夜灯塔')",
                boardEpisode, f.project(), episode);
        jdbc.update("INSERT INTO afv_storyboard_scene(id,storyboard_id,episode_id) VALUES(?,?,?)", boardScene, f.project(), boardEpisode);
        if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config WHERE is_default=TRUE AND deleted=FALSE", Long.class) == 0)
            jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('智能体测试存储','local',?,TRUE)", FILES.toString());
        var png = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB), "png", png);
        var frame = intake.upload(new ByteArrayInputStream(png.toByteArray()), "已有首帧.png", "image/png", UUID.randomUUID().toString(), f.project(),
                ResourceCommandContext.user(actor(f), f.canvas()));
        var video = intake.upload(new ByteArrayInputStream(CanvasAgentMediaFixtures.video(FILES.resolve("organize-" + f.user()), 1)),
                "已有镜头.mp4", "video/mp4", UUID.randomUUID().toString(), f.project(), ResourceCommandContext.user(actor(f), f.canvas()));
        jdbc.update("INSERT INTO afv_storyboard_item(id,storyboard_id,storyboard_episode_id,storyboard_scene_id,content,video_url) VALUES(?,?,?,?,'镜头 2-1',?)",
                missing, f.project(), boardEpisode, boardScene, "media://" + video.id());
        jdbc.update("INSERT INTO afv_storyboard_item(id,storyboard_id,storyboard_episode_id,storyboard_scene_id,content,first_frame_image_url) VALUES(?,?,?,?,'镜头 2-2',?)",
                ready, f.project(), boardEpisode, boardScene, "media://" + frame.id());
        var scenario = new CanvasOrganizationScenario(mapper);
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call ->
                new AiProviderService.AgentScopeModelProvision(scenario, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        var started = pipeline.start(new AiChatReqVO().setModelId(model.getId()).setAgentType("canvas_assistant").setProjectId(f.project())
                .setCanvasContext(new CanvasAgentContext(f.canvas(), List.of(), List.of(), null, 1L)).setToolExecutionMode("ALWAYS_ALLOW")
                .setMessage("把第二集的剧本、角色、分镜和已有视频整理到画布，标出缺少首帧的镜头；只整理，不生成素材。"), f.user()).block(Duration.ofSeconds(15));
        await().atMost(Duration.ofSeconds(60)).pollInterval(Duration.ofMillis(200)).until(() ->
                Set.of("COMPLETED", "FAILED", "CANCELLED").contains(jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId())));
        assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId()))
                .withFailMessage("整理状态：%s，业务步数：%s，最后工具结果：%s",
                        jdbc.queryForMap("SELECT status,error_message FROM afv_agent_run WHERE run_id=?", started.runId()),
                        scenario.calls(), scenario.lastResult())
                .isEqualTo("COMPLETED");
        var graph = canvases.snapshot(f.canvas(), actor(f)).graph();
        assertThat(graph.nodes()).hasSize(9);
        assertThat(graph.edges()).hasSize(1);
        String group = scenario.created().path("idMappings").path("$group").asText();
        assertThat(graph.nodes().stream().filter(node -> !node.id().equals(group))).allSatisfy(node -> assertThat(node.layout().parentId()).isEqualTo(group));
        assertThat(graph.nodes().stream().filter(node -> node.resource() != null).map(node -> node.resource().id()).toList())
                .containsExactlyInAnyOrder(episode, scene, f.project(), boardEpisode, missing, ready);
        String marker = scenario.created().path("idMappings").path("$note").asText();
        String missingNode = scenario.created().path("idMappings").path("$missing").asText();
        assertThat(graph.edges().getFirst().sourceNodeId()).isEqualTo(missingNode);
        assertThat(graph.edges().getFirst().targetNodeId()).isEqualTo(marker);
        assertThat(graph.nodes().stream().filter(node -> node.id().equals(scenario.created().path("idMappings").path("$video").asText())))
                .singleElement().satisfies(node -> assertThat(node.config().path("resourceId").asText()).isEqualTo(video.id()));
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission WHERE project_id=?", Integer.class, f.project())).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_run WHERE canvas_id=?", Integer.class, f.canvas())).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE agent_run_id=?", Integer.class, started.runId())).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_resource WHERE project_id=? AND state='READY'", Integer.class, f.project())).isEqualTo(2);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_script_episode WHERE id=?", Integer.class, f.project())).isZero();
        assertThat(scenario.calls()).isEqualTo(17);
        Files.writeString(Path.of("target/g-organize-episode-journey-results.json"), mapper.writerWithDefaultPrettyPrinter().writeValueAsString(Map.of(
                "model", "scripted model through actual AgentScope SDK", "toolQueriesUseReturnedIdsAndVersions", true,
                "nodes", 9, "groups", 1, "missingFrameMarkers", 1, "existingMediaReady", 2, "modelRequests", scenario.calls(),
                "generationSubmissions", 0, "canvasRuns", 0, "businessMutations", 0)));
    }

    @Test void assistantRegeneratesOnlySelectedRainNightShotsPreservingAppearanceAndOldResults() throws Exception {
        var f = fixture(); professionalInputTree(f); var model = model(f);
        long second = IDS.incrementAndGet(), unselected = IDS.incrementAndGet(), otherEpisode = IDS.incrementAndGet();
        long otherScene = IDS.incrementAndGet(), sceneAsset = IDS.incrementAndGet(), sceneItem = IDS.incrementAndGet();
        long videoModel = IDS.incrementAndGet();
        jdbc.update("UPDATE afv_asset_item SET properties=JSON_OBJECT('description','蓝色雨衣、短发，保持既有造型') WHERE id=?", f.project());
        jdbc.update("INSERT INTO afv_asset(id,project_id,name,type,owner_type,owner_id,user_id) VALUES(?,?,'灯塔','scene',1,?,?)",
                sceneAsset, f.project(), f.user(), f.user());
        jdbc.update("INSERT INTO afv_asset_item(id,asset_id,name,item_type,properties) VALUES(?,?,'灯塔外景','original','{}')", sceneItem, sceneAsset);
        jdbc.update("INSERT INTO afv_storyboard_episode(id,storyboard_id,episode_number,title) VALUES(?,?,2,'保持原样的第二集')", otherEpisode, f.project());
        jdbc.update("INSERT INTO afv_storyboard_scene(id,storyboard_id,episode_id) VALUES(?,?,?)", otherScene, f.project(), otherEpisode);
        jdbc.update("INSERT INTO afv_storyboard_item(id,storyboard_id,storyboard_episode_id,storyboard_scene_id,content) VALUES(?,?,?,?,'白天的原始镜头')",
                second, f.project(), f.project(), f.project());
        jdbc.update("INSERT INTO afv_storyboard_item(id,storyboard_id,storyboard_episode_id,storyboard_scene_id,content) VALUES(?,?,?,?,'未选中的白天镜头')",
                unselected, f.project(), otherEpisode, otherScene);
        if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config WHERE is_default=TRUE AND deleted=FALSE", Long.class) == 0)
            jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('智能体测试存储','local',?,TRUE)", FILES.toString());
        var png = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(1280, 720, BufferedImage.TYPE_INT_RGB), "png", png);
        byte[] oldVideo = CanvasAgentMediaFixtures.video(FILES.resolve("selected-old-" + f.user()), 1);
        byte[] newVideo = CanvasAgentMediaFixtures.video(FILES.resolve("selected-new-" + f.user()), 4);
        var userContext = ResourceCommandContext.user(actor(f), f.canvas());
        var frame = intake.upload(new ByteArrayInputStream(png.toByteArray()), "原造型首帧.png", "image/png", UUID.randomUUID().toString(), f.project(), userContext);
        var original = intake.upload(new ByteArrayInputStream(oldVideo), "修改前视频.mp4", "video/mp4", UUID.randomUUID().toString(), f.project(), userContext);
        for (long shot : List.of(f.project(), second, unselected)) jdbc.update("""
                UPDATE afv_storyboard_item SET character_ids=JSON_ARRAY(?),scene_asset_item_id=?,first_frame_image_url=?,video_url=?,video_prompt='白天灯塔',duration=4 WHERE id=?
                """, f.project(), sceneItem, "media://" + frame.id(), "media://" + original.id(), shot);
        var inputs = mapper.createArrayNode();
        for (int index = 0; index < 3; index++) {
            long shot = List.of(f.project(), second, unselected).get(index);
            var op = nodes("$shot" + index, "镜头 " + index);
            var node = (ObjectNode) op.get(0).path("node"); node.put("kind", "shot").set("config", mapper.createObjectNode());
            node.set("resource", mapper.valueToTree(Map.of("type", "shot", "id", shot, "version", 0)));
            inputs.addAll(op);
        }
        var ids = mapper.valueToTree(mutations.apply(inputs, invocation(f, "selected-fixture"))).path("idMappings");
        var requests = Collections.synchronizedList(new ArrayList<byte[]>());
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/videos", exchange -> {
            String path = exchange.getRequestURI().getPath();
            byte[] bytes; String contentType = "application/json";
            if (exchange.getRequestMethod().equals("POST") && path.equals("/v1/videos")) {
                requests.add(exchange.getRequestBody().readAllBytes());
                bytes = mapper.writeValueAsBytes(Map.of("id", "selected_" + requests.size(), "status", "queued"));
            } else if (path.matches("/v1/videos/selected_[12]/content")) {
                boolean cover = "variant=thumbnail".equals(exchange.getRequestURI().getQuery());
                bytes = cover ? png.toByteArray() : newVideo; contentType = cover ? "image/png" : "video/mp4";
            } else if (path.matches("/v1/videos/selected_[12]")) {
                bytes = mapper.writeValueAsBytes(Map.of("id", path.substring(path.lastIndexOf('/') + 1), "status", "completed", "seconds", "4"));
            } else { exchange.sendResponseHeaders(404, -1); exchange.close(); return; }
            exchange.getResponseHeaders().add("Content-Type", contentType); exchange.sendResponseHeaders(200, bytes.length);
            try (var output = exchange.getResponseBody()) { output.write(bytes); }
        });
        server.start();
        try {
            jdbc.update("INSERT INTO afv_api_config(id,name,platform,video_protocol,api_url,api_key) VALUES(?,'选区验收本机视频服务','openai_compatible','openai',?,'test-key')",
                    videoModel, "http://127.0.0.1:" + server.getAddress().getPort());
            jdbc.update("""
                    INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels)
                    VALUES(?,'选区验收视频模型','sora-2',3,'openai',?,'{"supportFirstFrame":true,"referenceImageInputFormats":["data_uri"],"maxImageInputs":1,"supportedDurations":[4],"defaultDuration":4}',2,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())
                    """, videoModel, videoModel);
            var scenario = new CanvasSelectedShotsScenario(mapper, videoModel);
            when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call ->
                    new AiProviderService.AgentScopeModelProvision(scenario, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
            var started = pipeline.start(new AiChatReqVO().setModelId(model.getId()).setAgentType("canvas_assistant").setProjectId(f.project())
                    .setCanvasContext(new CanvasAgentContext(f.canvas(), List.of(ids.path("$shot0").asText(), ids.path("$shot1").asText()), List.of(), null,
                            canvases.snapshot(f.canvas(), actor(f)).lastSequence())).setToolExecutionMode("DEFAULT")
                    .setMessage("把选中的两个镜头改为雨夜，保持角色造型，重新生成视频；保留新旧视频用于比较，经我批准再采用，其他镜头和分集保持原样。"), f.user()).block(Duration.ofSeconds(15));
            for (int step : List.of(6, 7, 8, 10, 14, 15, 16)) {
                await().atMost(Duration.ofSeconds(40)).pollInterval(Duration.ofMillis(200)).until(() -> Set.of("WAITING_CONFIRMATION", "FAILED", "CANCELLED").contains(
                        jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId())));
                assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId()))
                        .withFailMessage("选区场景 step=%s，模型调用=%s，结果=%s，状态=%s", step, scenario.calls(), scenario.lastResult(),
                                jdbc.queryForMap("SELECT status,error_message FROM afv_agent_run WHERE run_id=?", started.runId())).isEqualTo("WAITING_CONFIRMATION");
                var approval = new ToolConfirmationReqVO(); approval.setRunId(started.runId());
                approval.setReplyId(jdbc.queryForObject("SELECT waiting_reply_id FROM afv_agent_run WHERE run_id=?", String.class, started.runId()));
                var pending = waitingStates.getPendingConfirmationAuthorized(started.runId(), f.user(), approval.getReplyId()).block(Duration.ofSeconds(10));
                assertThat(pending.decisionIds()).containsExactly("selected-step-" + step);
                var decision = new ToolConfirmationReqVO.DecisionVO(); decision.setToolCallId("selected-step-" + step); decision.setApproved(true);
                approval.setDecisions(List.of(decision)); confirmations.respond(approval, f.user()).block(Duration.ofSeconds(10));
            }
            await().atMost(Duration.ofSeconds(30)).until(() -> Set.of("COMPLETED", "FAILED", "CANCELLED").contains(
                    jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId())));
            assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId()))
                    .withFailMessage("最终结果：%s", scenario.lastResult()).isEqualTo("COMPLETED");
            assertThat(requests).hasSize(2).allSatisfy(request -> {
                assertThat(new String(request, StandardCharsets.UTF_8)).contains("雨夜", "蓝色雨衣", "input_reference");
                assertThat(new String(request, StandardCharsets.ISO_8859_1)).contains(new String(png.toByteArray(), StandardCharsets.ISO_8859_1));
            });
            var run = canvasRuns.view(scenario.runId(), actor(f));
            assertThat(run.cost().videoSeconds()).isEqualTo(8);
            assertThat(run.nodes()).hasSize(4).noneMatch(node -> node.nodeId().equals(ids.path("$shot2").asText()));
            for (int index = 0; index < 2; index++) {
                long shot = List.of(f.project(), second).get(index);
                var row = jdbc.queryForMap("SELECT content,video_url,version,character_ids,scene_asset_item_id,first_frame_image_url FROM afv_storyboard_item WHERE id=?", shot);
                assertThat(row).containsEntry("version", 2).containsEntry("scene_asset_item_id", sceneItem)
                        .containsEntry("first_frame_image_url", "media://" + frame.id())
                        .containsEntry("video_url", "media://" + scenario.outputs().get(index).path("mediaId").asText());
                assertThat(row.get("character_ids").toString()).contains(String.valueOf(f.project()));
                assertThat(row.get("content").toString()).contains("雨夜");
            }
            assertThat(jdbc.queryForMap("SELECT content,video_url,video_prompt,version FROM afv_storyboard_item WHERE id=?", unselected))
                    .containsEntry("content", "未选中的白天镜头").containsEntry("video_url", "media://" + original.id()).containsEntry("video_prompt", "白天灯塔").containsEntry("version", 0);
            assertThat(jdbc.queryForObject("SELECT version FROM afv_storyboard_episode WHERE id=?", Integer.class, otherEpisode)).isZero();
            assertThat(jdbc.queryForObject("SELECT state FROM afv_media_resource WHERE id=?", String.class, original.id())).isEqualTo("READY");
            assertThat(canvases.snapshot(f.canvas(), actor(f)).graph().nodes()).hasSize(9);
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE agent_run_id=?", Integer.class, started.runId())).isEqualTo(4);
            assertThat(scenario.calls()).isEqualTo(19);
            Files.writeString(Path.of("target/g-selected-shots-journey-results.json"), mapper.writerWithDefaultPrettyPrinter().writeValueAsString(Map.of(
                    "model", "scripted root model through actual AgentScope SDK", "selectedShots", 2, "unselectedShotsUnchanged", 1,
                    "otherEpisodeVersion", 0, "providerPosts", requests.size(), "videoBudgetSeconds", 8, "approvalSteps", 7,
                    "originalFrameBytesPreserved", true, "oldVideoReady", true, "modelRequests", scenario.calls())));
        } finally { server.stop(0); }
    }

    @Test void twoEditorsCanChangeDifferentResourcesWhileAssistantKeepsItsFrozenGenerationAndConflictsAreExplicit() throws Exception {
        var f = fixture();
        professionalInputTree(f);
        var model = model(f);
        long editor = IDS.incrementAndGet(), imageModel = IDS.incrementAndGet();
        jdbc.update("INSERT INTO sys_user(id,username,password) VALUES(?,?,'test')", editor, "concurrent-editor-" + editor);
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,3)", f.project(), editor);
        jdbc.update("UPDATE afv_script_scene_item SET dialogues=? WHERE id=?",
                mapper.writeValueAsString(List.of(Map.of("type", 1, "character_name", "阿青", "content", "原始对白：灯塔还亮着", "sortOrder", 0))), f.project());
        var png = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB), "png", png);
        byte[] response = mapper.writeValueAsBytes(Map.of("data", List.of(Map.of("b64_json", Base64.getEncoder().encodeToString(png.toByteArray())))));
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var providerCalls = new AtomicInteger();
        var providerPrompt = new AtomicReference<String>();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/images/generations", exchange -> {
            providerPrompt.set(mapper.readTree(exchange.getRequestBody().readAllBytes()).path("prompt").asText());
            providerCalls.incrementAndGet();
            entered.countDown();
            try {
                if (!release.await(45, TimeUnit.SECONDS)) throw new AssertionError("并发编辑未及时释放模拟供应商");
                exchange.getResponseHeaders().add("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, response.length);
                try (var output = exchange.getResponseBody()) { output.write(response); }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                exchange.close();
            }
        });
        server.start();
        try {
            jdbc.update("INSERT INTO afv_api_config(id,name,platform,image_protocol,api_url,api_key) VALUES(?,'并发验收本机供应商','openai_compatible','openai',?,'test-key')",
                    imageModel, "http://127.0.0.1:" + server.getAddress().getPort());
            jdbc.update("""
                    INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels)
                    VALUES(?,'并发验收本机图片模型','gpt-image-2',2,'openai',?,'{"defaultWidth":32,"defaultHeight":32}',1,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())
                    """, imageModel, imageModel);
            if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config WHERE is_default=TRUE AND deleted=FALSE", Long.class) == 0)
                jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('智能体测试存储','local',?,TRUE)", FILES.toString());
            var operations = nodes("$scene", "正在生成的场次");
            var scene = (ObjectNode) operations.get(0).path("node");
            scene.put("kind", "scriptScene").set("config", mapper.createObjectNode());
            scene.set("resource", mapper.valueToTree(Map.of("type", "scriptScene", "id", f.project(), "version", 0)));
            var shot = nodes("$shot", "另一位用户编辑的镜头");
            ((ObjectNode) shot.get(0).path("node")).put("kind", "shot").set("config", mapper.createObjectNode());
            ((ObjectNode) shot.get(0).path("node")).set("resource", mapper.valueToTree(Map.of("type", "shot", "id", f.project(), "version", 0)));
            operations.addAll(shot);
            var image = nodes("$image", "场次参考图");
            ((ObjectNode) image.get(0).path("node")).put("kind", "imageGeneration").set("config",
                    mapper.valueToTree(Map.of("modelId", imageModel, "width", 32, "height", 32, "count", 1)));
            operations.addAll(image);
            operations.addObject().put("type", "canvas.edge.create").set("edge", mapper.valueToTree(Map.of("id", "$scene-image", "kind", "execution",
                    "sourceNodeId", "$scene", "sourcePort", "text", "targetNodeId", "$image", "targetPort", "prompt", "inputIndex", 0, "version", 0)));
            var scenario = new CanvasConcurrentScenario(mapper, operations);
            when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call ->
                    new AiProviderService.AgentScopeModelProvision(scenario, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
            var started = pipeline.start(new AiChatReqVO().setModelId(model.getId()).setAgentType("canvas_assistant").setProjectId(f.project())
                    .setCanvasContext(new CanvasAgentContext(f.canvas(), List.of(), List.of(), null, 1L)).setToolExecutionMode("ALWAYS_ALLOW")
                    .setMessage("根据场次对白生成一张参考图，期间允许协作者继续编辑；保留新版本，不自动写回生成结果。"), f.user()).block(Duration.ofSeconds(15));
            await().atMost(Duration.ofSeconds(30)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                    "SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId())).isEqualTo("WAITING_CONFIRMATION"));
            var approval = new ToolConfirmationReqVO();
            approval.setRunId(started.runId());
            approval.setReplyId(jdbc.queryForObject("SELECT waiting_reply_id FROM afv_agent_run WHERE run_id=?", String.class, started.runId()));
            var pending = waitingStates.getPendingConfirmationAuthorized(started.runId(), f.user(), approval.getReplyId()).block(Duration.ofSeconds(10));
            assertThat(pending.decisionIds()).containsExactly("concurrent-step-3");
            var decision = new ToolConfirmationReqVO.DecisionVO(); decision.setToolCallId("concurrent-step-3"); decision.setApproved(true);
            approval.setDecisions(List.of(decision));
            confirmations.respond(approval, f.user()).block(Duration.ofSeconds(10));
            assertThat(entered.await(20, TimeUnit.SECONDS)).isTrue();
            await().atMost(Duration.ofSeconds(10)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                    "SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId())).isEqualTo("WAITING_EXTERNAL"));
            long cursor = canvases.snapshot(f.canvas(), actor(f)).lastSequence();
            var firstHeaders = new HttpHeaders();
            firstHeaders.setBearerAuth(tokens.createToken(f.user(), "canvas-agent-" + f.user(), null).getAccessToken());
            firstHeaders.setContentType(MediaType.APPLICATION_JSON);
            var secondHeaders = new HttpHeaders();
            secondHeaders.setBearerAuth(tokens.createToken(editor, "concurrent-editor-" + editor, null).getAccessToken());
            secondHeaders.setContentType(MediaType.APPLICATION_JSON);
            String endpoint = "/api/canvases/" + f.canvas() + "/resource-commands";
            var dialoguePatch = new ResourcePatchCommand("concurrent-dialogue", CanvasValueType.SCRIPT_SCENE, f.project(), 0,
                    mapper.valueToTree(Map.of("dialogues", List.of(Map.of("type", 1, "character_name", "阿青", "content", "用户 A 新对白：我会回来", "sortOrder", 0)))));
            var shotPatch = new ResourcePatchCommand("concurrent-shot", CanvasValueType.SHOT, f.project(), 0,
                    mapper.valueToTree(Map.of("content", "用户 B 新镜头：缓慢推近")));
            var a = CompletableFuture.supplyAsync(() -> http.postForEntity(endpoint, new HttpEntity<>(dialoguePatch, firstHeaders), JsonNode.class));
            var b = CompletableFuture.supplyAsync(() -> http.postForEntity(endpoint, new HttpEntity<>(shotPatch, secondHeaders), JsonNode.class));
            assertThat(a.join().getStatusCode()).withFailMessage("对白修改：%s", a.join().getBody()).isEqualTo(HttpStatus.OK);
            assertThat(b.join().getStatusCode()).withFailMessage("镜头修改：%s", b.join().getBody()).isEqualTo(HttpStatus.OK);
            var stale = new ResourcePatchCommand("concurrent-stale", CanvasValueType.SCRIPT_SCENE, f.project(), 0, dialoguePatch.patch());
            assertThat(http.postForEntity(endpoint, new HttpEntity<>(stale, secondHeaders), JsonNode.class).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
            for (int index = 0; index < 50 && resourceEvents.publishNext(); index++) { }
            var firstEvents = canvases.events(f.canvas(), actor(f), cursor, 100).events();
            var secondEvents = canvases.events(f.canvas(), new CanvasActor(editor, null), cursor, 100).events();
            assertThat(firstEvents).isEqualTo(secondEvents);
            assertThat(firstEvents.stream().filter(event -> event.type().equals("resource.changed"))
                    .map(event -> event.payload().path("resourceType").asText()).toList()).contains("scriptScene", "shot");
            assertThat(providerPrompt.get()).contains("原始对白：灯塔还亮着").doesNotContain("用户 A 新对白");
            assertThat(providerCalls).hasValue(1);
            assertThat(scenario.calls()).isEqualTo(5);
            assertThat(jdbc.queryForObject("SELECT owner_instance_id FROM afv_agent_run WHERE run_id=?", String.class, started.runId())).isNull();
            release.countDown();
            await().atMost(Duration.ofSeconds(30)).until(() -> Set.of("COMPLETED", "FAILED", "CANCELLED").contains(
                    jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId())));
            assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId()))
                    .withFailMessage("并发场景：%s / %s", scenario.lastResult(), jdbc.queryForMap("SELECT status,error_message FROM afv_agent_run WHERE run_id=?", started.runId()))
                    .isEqualTo("COMPLETED");
            var completed = canvasRuns.view(scenario.runId(), actor(f));
            var generated = completed.nodes().stream().filter(node -> node.nodeId().equals(scenario.created().path("idMappings").path("$image").asText())).findFirst().orElseThrow();
            assertThat(completed.status()).isEqualTo(CanvasRunData.State.SUCCEEDED);
            assertThat(generated.derivedState()).isEqualTo("STALE");
            assertThat(generated.inputs().at("prompt").getFirst().text()).contains("原始对白：灯塔还亮着").doesNotContain("用户 A 新对白");
            assertThat(jdbc.queryForObject("SELECT state FROM afv_media_resource WHERE id=?", String.class, generated.outputs().at("images").getFirst().mediaId())).isEqualTo("READY");
            assertThat(jdbc.queryForObject("SELECT dialogues FROM afv_script_scene_item WHERE id=?", String.class, f.project())).contains("用户 A 新对白：我会回来");
            assertThat(jdbc.queryForObject("SELECT content FROM afv_storyboard_item WHERE id=?", String.class, f.project())).isEqualTo("用户 B 新镜头：缓慢推近");
            assertThat(jdbc.queryForObject("SELECT version FROM afv_script_scene_item WHERE id=?", Integer.class, f.project())).isEqualTo(1);
            assertThat(jdbc.queryForObject("SELECT version FROM afv_storyboard_item WHERE id=?", Integer.class, f.project())).isEqualTo(1);
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE agent_run_id=?", Integer.class, started.runId())).isZero();
            assertThat(providerCalls).hasValue(1);
            assertThat(scenario.calls()).isEqualTo(7);
            Files.writeString(Path.of("target/g-concurrent-editors-journey-results.json"), mapper.writerWithDefaultPrettyPrinter().writeValueAsString(Map.of(
                    "environment", "isolated Testcontainers and local HTTP image provider", "independentEditors", 2,
                    "sameFieldConflictHttpStatus", 409, "sameEventsVisibleToBothEditors", true, "frozenOriginalDialogue", true,
                    "generatedResultState", generated.derivedState(), "providerRequests", providerCalls.get(), "modelRequests", scenario.calls(),
                    "agentBusinessWrites", 0, "latestHumanVersionsPreserved", true)));
        } finally {
            release.countDown();
            server.stop(0);
        }
    }

    @ParameterizedTest @ValueSource(strings = {"length", "tool_calls"})
    void invalidModelOutputFailsBeforeApprovalAndKeepsTheOriginalRequestUsage(String finishReason) throws Exception {
        var f = fixture();
        var model = model(f);
        var calls = new AtomicInteger();
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                new ChatModelBase() {
                    @Override public String getModelName() { return "malformed-canvas-arguments"; }
                    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
                        calls.incrementAndGet();
                        return Flux.just(ChatResponse.builder().id("broken-output")
                                .content(List.of(new ToolUseBlock("broken-call", "apply_canvas_operations", Map.of(), "{\"operations\":[", null)))
                                .usage(ChatUsage.builder().inputTokens(10).outputTokens(2048).build())
                                .finishReason(finishReason).build());
                    }
                }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        var request = new AiChatReqVO().setConversationId(UUID.randomUUID().toString()).setModelId(model.getId())
                .setAgentType("canvas_assistant").setProjectId(f.project()).setToolExecutionMode("DEFAULT")
                .setCanvasContext(new CanvasAgentContext(f.canvas(), List.of(), List.of(), null, 1L))
                .setMessage("创建一个文本节点");
        var started = pipeline.start(request, f.user()).block(Duration.ofSeconds(15));
        await().atMost(Duration.ofSeconds(30)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT status FROM afv_agent_run WHERE run_id=?", String.class, started.runId())).isEqualTo("FAILED"));
        assertThat(calls).hasValue(1);
        assertThat(canvases.snapshot(f.canvas(), actor(f)).graph().nodes()).isEmpty();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_canvas_operation WHERE agent_run_id=?", Long.class, started.runId())).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_agent_model_request WHERE run_id=? AND status='COMPLETED' AND permit_held=FALSE",
                Long.class, started.runId())).isEqualTo(1);
        assertThat(jdbc.queryForObject("SELECT output_tokens FROM afv_agent_model_call_usage WHERE run_id=?", Long.class, started.runId())).isEqualTo(2048);
    }

    @Test void normalMessageAfterCancellationRestoresRootHistoryOnceAndExcludesChildInstructions() throws Exception {
        var f = fixture();
        var model = model(f);
        String conversation = UUID.randomUUID().toString();
        String firstPrompt = "记住灯塔暗号 " + UUID.randomUUID() + "，准备一个文本节点。";
        var operations = nodes("$pending", "尚未获批的节点");
        var calls = new AtomicInteger();
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                new ChatModelBase() {
                    @Override public String getModelName() { return "conversation-generation-recovery"; }
                    @Override protected Flux<ChatResponse> doStream(List<Msg> input, List<ToolSchema> schema, GenerateOptions options) {
                        int step = calls.getAndIncrement();
                        if (step == 0) {
                            Map<String, Object> arguments = Map.of("operations", mapper.convertValue(operations, List.class));
                            return Flux.just(ChatResponse.builder().id("prepare-node").content(List.of(ToolUseBlock.builder()
                                    .id("pending-create").name("apply_canvas_operations").input(arguments)
                                    .content(new CanvasJson(mapper).write(arguments)).build())).finishReason("tool_calls").build());
                        }
                        assertThat(input.stream().filter(message -> message.getRole() == MsgRole.USER
                                && message.getTextContent().contains(firstPrompt))).hasSize(1);
                        assertThat(input.stream().map(Msg::getTextContent).toList().toString())
                                .contains("apply_canvas_operations", "cancelled")
                                .doesNotContain("仅供子任务使用的内部提示");
                        assertThat(input.getLast().getTextContent()).contains(step == 1 ? "继续检查原任务" : "再次确认暗号");
                        return Flux.just(ChatResponse.builder().id("answer-" + step)
                                .content(List.of(TextBlock.builder().text("记得暗号，未执行已经取消的建图请求。").build()))
                                .finishReason("stop").build());
                    }
                }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        var request = new AiChatReqVO().setConversationId(conversation).setModelId(model.getId())
                .setAgentType("canvas_assistant").setProjectId(f.project()).setToolExecutionMode("DEFAULT")
                .setCanvasContext(new CanvasAgentContext(f.canvas(), List.of(), List.of(), null, 1L)).setMessage(firstPrompt);
        var first = pipeline.start(request, f.user()).block(Duration.ofSeconds(15));
        await().atMost(Duration.ofSeconds(30)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT status FROM afv_agent_run WHERE run_id=?", String.class, first.runId())).isEqualTo("WAITING_CONFIRMATION"));
        var headers = new HttpHeaders();
        headers.setBearerAuth(tokens.createToken(f.user(), "canvas-agent-" + f.user(), null).getAccessToken());
        var cancelled = http.postForEntity("/api/ai/pipeline/cancel?runId=" + first.runId(), new HttpEntity<>(headers), JsonNode.class);
        assertThat(cancelled.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(cancelled.getBody().path("code").asInt()).isZero();
        await().atMost(Duration.ofSeconds(30)).untilAsserted(() -> {
            assertThat(jdbc.queryForObject("SELECT status FROM afv_agent_run WHERE run_id=?", String.class, first.runId())).isEqualTo("CANCELLED");
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_agent_message WHERE run_id=? AND tool_status='cancelled'", Long.class, first.runId())).isPositive();
        });
        // Reproduce a historical child prompt without parent_tool_call_id; the run identity must still exclude it.
        jdbc.update("UPDATE afv_agent_run SET conversation_id=?,parent_run_id=?,parent_tool_call_id='child-history',agent_name='child-history',status='COMPLETED' WHERE run_id=?",
                conversation, first.runId(), f.runId());
        messageAllocator.append(conversation, AgentMessage.builder().runId(f.runId()).role("user").content("仅供子任务使用的内部提示").build());
        var second = pipeline.start(request.setMessage("继续检查原任务，不要执行已取消的建图请求。"), f.user()).block(Duration.ofSeconds(15));
        await().atMost(Duration.ofSeconds(30)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT status FROM afv_agent_run WHERE run_id=?", String.class, second.runId())).isEqualTo("COMPLETED"));
        assertThat(second.agentStateSessionId()).isNotEqualTo(first.agentStateSessionId());
        var third = pipeline.start(request.setMessage("再次确认暗号，保持现有画布不变。"), f.user()).block(Duration.ofSeconds(15));
        await().atMost(Duration.ofSeconds(30)).untilAsserted(() -> assertThat(jdbc.queryForObject(
                "SELECT status FROM afv_agent_run WHERE run_id=?", String.class, third.runId())).isEqualTo("COMPLETED"));
        assertThat(third.agentStateSessionId()).isEqualTo(second.agentStateSessionId());
        assertThat(calls).hasValue(3);
        assertThat(canvases.snapshot(f.canvas(), actor(f)).graph().nodes()).isEmpty();
    }

    @ParameterizedTest @ValueSource(ints = {1, 2})
    void professionalGenerationUsesSelectedModelAndStopsAtTheCumulativeImageBudget(int limit) throws Exception {
        var f = fixture();
        var model = model(f);
        var png = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB), "png", png);
        byte[] result = mapper.writeValueAsBytes(Map.of("data", List.of(Map.of("b64_json", Base64.getEncoder().encodeToString(png.toByteArray())))));
        var calls = new AtomicInteger();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/images/generations", exchange -> {
            exchange.getRequestBody().readAllBytes(); calls.incrementAndGet();
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, result.length);
            try (var output = exchange.getResponseBody()) { output.write(result); }
        });
        server.start();
        try {
            long imageModel = IDS.incrementAndGet();
            jdbc.update("INSERT INTO afv_api_config(id,name,platform,image_protocol,api_url,api_key) VALUES(?,'专业节点图片供应商','openai_compatible','openai',?,'test-key')",
                    imageModel, "http://127.0.0.1:" + server.getAddress().getPort());
            jdbc.update("""
                    INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels)
                    VALUES(?,'专业节点选定图片模型','gpt-image-2',2,'openai',?,'{"defaultWidth":64,"defaultHeight":64}',1,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())
                    """, imageModel, imageModel);
            if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config WHERE is_default=TRUE AND deleted=FALSE", Long.class) == 0)
                jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('智能体测试存储','local',?,TRUE)", FILES.toString());
            var steps = new AtomicInteger();
            var submission = new AtomicReference<String>();
            var finishSummary = new CompletableFuture<Void>();
            when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                    new ChatModelBase() {
                        @Override public String getModelName() { return "professional-image-contract"; }
                        @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
                            int step = steps.getAndIncrement();
                            if (step > 0 && step <= limit) {
                                var accepted = lastTool(messages, "generate_image");
                                assertThat(accepted.path("status").asText()).withFailMessage("生成受理：%s", accepted).isEqualTo("pending");
                                submission.set(accepted.path("submissionId").asText());
                            }
                            if (step == limit + 1) {
                                assertThat(new CanvasJson(mapper).write(messages)).contains("数量上限");
                            }
                            if (step > limit + 1) {
                                var completed = lastTool(messages, "await_generation_task");
                                assertThat(completed.path("state").asText()).withFailMessage("生成完成：%s", completed).isEqualTo("SUCCEEDED");
                                return Mono.fromFuture(finishSummary).thenMany(Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder()
                                        .text("已完成约定数量图片；额外请求因达到上限未提交").build())).finishReason("stop").build()));
                            }
                            String tool = step <= limit ? "generate_image" : "await_generation_task";
                            Map<String, Object> args = step <= limit ? Map.of("prompt", "雨夜古城第" + step + "张") : Map.of("submissionId", submission.get());
                            return Flux.just(ChatResponse.builder().content(List.of(ToolUseBlock.builder().id("gen-" + step).name(tool)
                                    .input(args).content(new CanvasJson(mapper).write(args)).build())).finishReason("tool_calls").build());
                        }
                    }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
            var input = nodes("$agent", "受预算约束的图片创作");
            ((ObjectNode) input.get(0).path("node")).put("kind", "agent").set("config", mapper.valueToTree(Map.of(
                    "agentType", "ai_media", "modelId", model.getId(), "inputType", "text", "resultType", "image", "instruction", "生成雨夜古城",
                    "imageModelId", imageModel, "maxImages", limit, "maxVideoSeconds", 0, "maxAgentRuns", 1)));
            var ids = mapper.valueToTree(mutations.apply(input, invocation(f, "image-budget-node"))).path("idMappings");
            var plan = canvasRuns.plan(f.canvas(), new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(), CanvasRunData.Scope.SINGLE,
                    List.of(ids.path("$agent").asText()), Map.of(), new CanvasRunData.Budget(1, limit, 0, 1, 1), new GenerationClientScope(f.user(), null)), actor(f));
            assertThat(plan.issues()).isEmpty();
            var run = canvasRuns.start(f.canvas(), new CanvasRunData.Start("professional-images", plan.id(), plan.hash(), new GenerationClientScope(f.user(), null)),
                    ResourceCommandContext.user(actor(f), f.canvas()));
            String nodeRun = run.nodes().getFirst().id();
            try {
                await().atMost(Duration.ofSeconds(30)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
                    var current = canvasRuns.view(run.id(), actor(f)).nodes().getFirst();
                    assertThat(current.status()).isEqualTo(CanvasRunData.State.RUNNING);
                    assertThat(current.outputs().at("results")).hasSize(limit).allMatch(value -> value.mediaId() != null);
                    assertThat(agentDetails.detail(run.id(), nodeRun, actor(f)).tasks()).allMatch(task -> task.confirmation() == null);
                });
            } finally { finishSummary.complete(null); }
            await().atMost(Duration.ofSeconds(30)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
                var current = canvasRuns.view(run.id(), actor(f));
                assertThat(current.status()).withFailMessage("专业图片生成：%s", current).isEqualTo(CanvasRunData.State.SUCCEEDED);
                assertThat(current.nodes().getFirst().outputs().at("results")).hasSize(limit).allSatisfy(output -> assertThat(output.mediaId()).isNotBlank());
            });
            assertThat(calls).hasValue(limit);
            assertAutomaticNodeTasks(nodeRun);
            assertThat(agentDetails.detail(run.id(), nodeRun, actor(f)).generations()).hasSize(limit)
                    .allSatisfy(generation -> {
                        assertThat(generation.state()).isEqualTo("SUCCEEDED");
                        assertThat(generation.modelName()).isNotBlank();
                        assertThat(generation.id()).isNotBlank();
                    });
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_generation_submission WHERE canvas_node_run_id=? AND JSON_EXTRACT(model_snapshot_json,'$.id')=?", Long.class, nodeRun, imageModel)).isEqualTo(limit);
        } finally { server.stop(0); }
    }

    private JsonNode lastTool(List<Msg> messages, String name) {
        var results = messages.stream().flatMap(message -> message.getContent().stream()).filter(ToolResultBlock.class::isInstance)
                .map(ToolResultBlock.class::cast).filter(block -> block.getName().equals(name)).toList();
        assertThat(results).isNotEmpty();
        String text = results.getLast().getOutput().stream().filter(TextBlock.class::isInstance).map(TextBlock.class::cast).map(TextBlock::getText).reduce("", String::concat);
        return new CanvasJson(mapper).read(text, JsonNode.class);
    }

    @ParameterizedTest
    @CsvSource({"asset_image_executor,assetItem,assetItem,assetItem,asset", "storyboard_frame_executor,shot,shot,shot,first",
            "storyboard_frame_executor,shot,shot,shot,last", "storyboard_video_executor,shot,shot,shot,video",
            "asset_image_gen,assetItem,assetItem,assetItem,asset", "storyboard_frame_gen,shot,shot,shot,first",
            "storyboard_video_gen,shot,shot,shot,video"})
    void professionalMediaExecutorsGenerateWaitAndAutomaticallyBindRealBusinessMedia(String profile, String kind, String port, String resultType, String frame) throws Exception {
        var f = fixture(); professionalInputTree(f); var model = model(f);
        boolean video = frame.equals("video");
        boolean delegated = profile.endsWith("_gen");
        var png = new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB), "png", png);
        byte[] result = mapper.writeValueAsBytes(Map.of("data", List.of(Map.of("b64_json", Base64.getEncoder().encodeToString(png.toByteArray())))));
        var calls = new AtomicInteger();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/images/generations", exchange -> {
            exchange.getRequestBody().readAllBytes(); calls.incrementAndGet();
            exchange.getResponseHeaders().add("Content-Type", "application/json"); exchange.sendResponseHeaders(200, result.length);
            try (var output = exchange.getResponseBody()) { output.write(result); }
        });
        if (video) {
            byte[] videoBytes = CanvasAgentMediaFixtures.video(FILES.resolve("video-business-" + f.user()), 4);
            server.createContext("/v1/videos", exchange -> {
                String path = exchange.getRequestURI().getPath();
                byte[] body; String contentType = "application/json";
                if (exchange.getRequestMethod().equals("POST") && path.equals("/v1/videos")) {
                    exchange.getRequestBody().readAllBytes(); calls.incrementAndGet();
                    body = mapper.writeValueAsBytes(Map.of("id", "business_video", "status", "queued"));
                } else if (path.equals("/v1/videos/business_video")) {
                    body = mapper.writeValueAsBytes(Map.of("id", "business_video", "status", "completed", "seconds", "4"));
                } else if (path.equals("/v1/videos/business_video/content")) {
                    boolean cover = "variant=thumbnail".equals(exchange.getRequestURI().getQuery());
                    body = cover ? png.toByteArray() : videoBytes; contentType = cover ? "image/png" : "video/mp4";
                } else { exchange.sendResponseHeaders(404, -1); exchange.close(); return; }
                exchange.getResponseHeaders().add("Content-Type", contentType); exchange.sendResponseHeaders(200, body.length);
                try (var output = exchange.getResponseBody()) { output.write(body); }
            });
        }
        server.start();
        try {
            long imageModel = IDS.incrementAndGet();
            jdbc.update("INSERT INTO afv_api_config(id,name,platform,image_protocol,video_protocol,api_url,api_key) VALUES(?,'正式回填验收','openai_compatible','openai','openai',?,'test-key')",
                    imageModel, "http://127.0.0.1:" + server.getAddress().getPort());
            jdbc.update("""
                    INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels)
                    VALUES(?,'业务生成媒体模型',?,?,'openai',?, ?,1,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())
                    """, imageModel, video ? "sora-2" : "gpt-image-2", video ? 3 : 2, imageModel, video ? "{}" : "{\"defaultWidth\":64,\"defaultHeight\":64}");
            if (jdbc.queryForObject("SELECT COUNT(*) FROM afv_storage_config WHERE is_default=TRUE AND deleted=FALSE", Long.class) == 0)
                jdbc.update("INSERT INTO afv_storage_config(name,type,base_path,is_default) VALUES('智能体测试存储','local',?,TRUE)", FILES.toString());
            var steps = new AtomicInteger(); var parentSteps = new AtomicInteger(); var reference = new AtomicReference<String>();
            String writeTool = frame.equals("asset") ? "update_asset_image" : video ? "update_storyboard_item_video" : "update_storyboard_item_frame";
            String generateTool = video ? "generate_video" : "generate_image";
            String childTool = frame.equals("asset") ? "generate_asset_image" : video ? "generate_storyboard_video" : "generate_storyboard_frame";
            when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                    new ChatModelBase() {
                        @Override public String getModelName() { return "professional-media-business"; }
                        @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
                            if (delegated && schema.stream().anyMatch(item -> item.getName().equals(childTool))) {
                                if (parentSteps.getAndIncrement() == 0) {
                                    Map<String, Object> delegation = Map.of("message", "生成并回填已选业务对象\nprojectId: " + f.project() + "\nassetId: " + f.project()
                                            + "\nitemId: " + f.project() + "\nstoryboardItemId: " + f.project() + "\nframeType: " + frame
                                            + "\nframePrompt: 雨夜灯塔\nvideoPrompt: 雨夜灯塔\nduration: 4");
                                    return Flux.just(ChatResponse.builder().content(List.of(ToolUseBlock.builder().id("business-child").name(childTool)
                                            .input(delegation).content(new CanvasJson(mapper).write(delegation)).build())).finishReason("tool_calls").build());
                                }
                                assertThat(new CanvasJson(mapper).write(messages)).contains("媒体已生成并保存到确认的业务对象");
                                return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("已收到子任务的真实业务回填结果").build())).finishReason("stop").build());
                            }
                            int step = steps.getAndIncrement(); String tool; String id; Map<String, Object> args;
                            if (step == 0) {
                                tool = generateTool; id = "business-generate"; args = video ? Map.of("prompt", "雨夜灯塔", "duration", 4) : Map.of("prompt", "雨夜灯塔 " + frame);
                            } else if (step == 1) {
                                var accepted = lastTool(messages, generateTool);
                                assertThat(accepted.path("status").asText()).as(accepted.toString()).isEqualTo("pending");
                                tool = "await_generation_task"; id = "business-wait"; args = Map.of("submissionId", accepted.path("submissionId").asText());
                            } else if (step == 2) {
                                var completed = lastTool(messages, "await_generation_task");
                                assertThat(completed.path("state").asText()).as(completed.toString()).isEqualTo("SUCCEEDED");
                                reference.set(completed.at(video ? "/outputs/0/videoUrl" : "/outputs/0/imageUrl").asText());
                                assertThat(reference.get()).startsWith("media://");
                                tool = writeTool; id = "business-bind";
                                args = frame.equals("asset") ? Map.of("assetId", f.project(), "itemId", f.project(), "expectedVersion", 0,
                                        "expectedParentVersion", 0, "imageUrl", reference.get(), "aiPrompt", "雨夜灯塔")
                                        : video ? Map.of("storyboardItemId", f.project(), "expectedVersion", 0, "videoUrl", reference.get(), "videoPrompt", "雨夜灯塔")
                                        : Map.of("storyboardItemId", f.project(), "expectedVersion", 0, "frameType", frame,
                                        "imageUrl", reference.get(), "framePrompt", "雨夜灯塔");
                            } else {
                                var saved = lastTool(messages, writeTool);
                                assertThat(saved.path("status").asText()).as(saved.toString()).isEqualTo("success");
                                return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("媒体已生成并保存到确认的业务对象").build())).finishReason("stop").build());
                            }
                            assertThat(schema).anyMatch(item -> item.getName().equals(tool));
                            return Flux.just(ChatResponse.builder().content(List.of(ToolUseBlock.builder().id(id).name(tool)
                                    .input(args).content(new CanvasJson(mapper).write(args)).build())).finishReason("tool_calls").build());
                        }
                    }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
            var input = nodes("$source", "待生成业务对象");
            var source = (ObjectNode) input.get(0).path("node"); source.put("kind", kind).set("config", mapper.createObjectNode());
            source.set("resource", mapper.valueToTree(Map.of("type", kind, "id", f.project(), "version", 0)));
            var agent = nodes("$execute", "生成并正式回填");
            ((ObjectNode) agent.get(0).path("node")).put("kind", "agent").set("config", mapper.valueToTree(Map.of(
                    "agentType", profile, "modelId", model.getId(), "inputType", kind, "resultType", resultType,
                    "instruction", "生成并回填 " + frame, video ? "videoModelId" : "imageModelId", imageModel, "maxImages", video ? 0 : 1, "maxVideoSeconds", video ? 4 : 0, "maxAgentRuns", delegated ? 2 : 1)));
            input.addAll(agent);
            input.addObject().put("type", "canvas.edge.create").set("edge", mapper.valueToTree(Map.of("id", "$business-input", "kind", "execution",
                    "sourceNodeId", "$source", "sourcePort", port, "targetNodeId", "$execute", "targetPort", "inputs", "inputIndex", 0, "version", 0)));
            var ids = mapper.valueToTree(mutations.apply(input, invocation(f, "image-business-create"))).path("idMappings");
            var plan = canvasRuns.plan(f.canvas(), new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(), CanvasRunData.Scope.DEPENDENCIES,
                    List.of(ids.path("$execute").asText()), Map.of(), new CanvasRunData.Budget(delegated ? 3 : 2, video ? 0 : 1, video ? 4 : 0, 1, 1), new GenerationClientScope(f.user(), null)), actor(f));
            assertThat(plan.issues()).isEmpty();
            var run = canvasRuns.start(f.canvas(), new CanvasRunData.Start("image-business", plan.id(), plan.hash(), new GenerationClientScope(f.user(), null)),
                    ResourceCommandContext.user(actor(f), f.canvas()));
            String nodeRun = run.nodes().stream().filter(node -> node.nodeId().equals(ids.path("$execute").asText())).findFirst().orElseThrow().id();
            await().atMost(Duration.ofSeconds(30)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
                var current = canvasRuns.view(run.id(), actor(f));
                assertThat(current.status()).withFailMessage("业务媒体回填：%s", current).isEqualTo(CanvasRunData.State.SUCCEEDED);
            });
            String column = frame.equals("asset") ? "image_url" : video ? "generated_video_url" : frame + "_frame_image_url";
            String table = frame.equals("asset") ? "afv_asset_item" : "afv_storyboard_item";
            assertThat(jdbc.queryForObject("SELECT " + column + " FROM " + table + " WHERE id=?", String.class, f.project())).isEqualTo(reference.get());
            assertThat(jdbc.queryForObject("SELECT version FROM " + table + " WHERE id=?", Integer.class, f.project())).isEqualTo(1);
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command c JOIN afv_agent_run r ON r.run_id=c.agent_run_id WHERE JSON_UNQUOTE(JSON_EXTRACT(r.canvas_scope_json,'$.nodeRunId'))=? AND c.tool_call_id='business-bind'", Long.class, nodeRun)).isEqualTo(1);
            assertThat(calls).hasValue(1); assertThat(steps).hasValue(4);
            assertThat(parentSteps).hasValue(delegated ? 2 : 0);
            assertAutomaticNodeTasks(nodeRun);
            assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_agent_run WHERE JSON_UNQUOTE(JSON_EXTRACT(canvas_scope_json,'$.nodeRunId'))=? AND status='COMPLETED'", Long.class, nodeRun)).isEqualTo(delegated ? 2 : 1);
        } finally { server.stop(0); }
    }

    @ParameterizedTest
    @CsvSource({"canvas_asset_extractor,scriptScene,scene,asset,1", "episode_storyboard_writer,scriptEpisode,episode,shot,2"})
    void professionalCreatorsPublishOnlyTheAssetsAndShotsActuallyCommittedThroughTools(String profile, String kind, String port, String resultType, int count) throws Exception {
        var f = fixture(); professionalInputTree(f); var model = model(f);
        boolean asset = resultType.equals("asset");
        String tool = asset ? "batch_create_assets" : "save_storyboard_scene_shots";
        Map<String, Object> args = asset ? Map.of("projectId", f.project(), "reuseExisting", false, "assets", List.of(Map.of(
                "type", "scene", "name", "雨夜灯塔", "description", "海边石砌灯塔，铜门和螺旋楼梯", "properties", Map.of(),
                "initialItem", Map.of("name", "雨夜灯塔外景", "properties", Map.of()))))
                : Map.of("storyboardId", f.project(), "storyboardEpisodeId", f.project(), "expectedParentVersion", 0,
                "sceneNumber", "1-2", "sceneHeading", "外景 灯塔 夜", "shots", List.of(Map.of("content", "阿青推开灯塔铜门", "duration", 3),
                Map.of("content", "特写信封上的日期", "duration", 2)));
        var calls = new AtomicInteger();
        when(providers.provisionSingleRequestModel(any(AiModel.class))).thenAnswer(call -> new AiProviderService.AgentScopeModelProvision(
                new ChatModelBase() {
                    @Override public String getModelName() { return "professional-business-creator"; }
                    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
                        assertThat(schema).anyMatch(item -> item.getName().equals(tool));
                        if (calls.getAndIncrement() == 0) return Flux.just(ChatResponse.builder().content(List.of(ToolUseBlock.builder()
                                .id("business-create").name(tool).input(args).content(new CanvasJson(mapper).write(args)).build())).finishReason("tool_calls").build());
                        var saved = lastTool(messages, tool);
                        assertThat(saved.path("status").asText()).as(saved.toString()).isEqualTo("success");
                        return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder().text("创作内容已经写入正式项目").build())).finishReason("stop").build());
                    }
                }, AgentKernelKey.modelRecordFingerprint(call.getArgument(0))));
        var input = nodes("$source", "创作依据");
        var source = (ObjectNode) input.get(0).path("node"); source.put("kind", kind).set("config", mapper.createObjectNode());
        source.set("resource", mapper.valueToTree(Map.of("type", kind, "id", f.project(), "version", 0)));
        var agent = nodes("$create", "创作正式内容");
        ((ObjectNode) agent.get(0).path("node")).put("kind", "agent").set("config", mapper.valueToTree(Map.of(
                "agentType", profile, "modelId", model.getId(), "inputType", kind, "resultType", resultType,
                "instruction", "创作雨夜灯塔的素材与镜头", "maxAgentRuns", 1)));
        input.addAll(agent);
        input.addObject().put("type", "canvas.edge.create").set("edge", mapper.valueToTree(Map.of("id", "$business-source", "kind", "execution",
                "sourceNodeId", "$source", "sourcePort", port, "targetNodeId", "$create", "targetPort", "inputs", "inputIndex", 0, "version", 0)));
        var ids = mapper.valueToTree(mutations.apply(input, invocation(f, "business-creation-plan"))).path("idMappings");
        var plan = canvasRuns.plan(f.canvas(), new CanvasRunData.PlanRequest(canvases.snapshot(f.canvas(), actor(f)).document().revision(), CanvasRunData.Scope.DEPENDENCIES,
                List.of(ids.path("$create").asText()), Map.of(), new CanvasRunData.Budget(2, 0, 0, 1, 1), new GenerationClientScope(f.user(), null)), actor(f));
        assertThat(plan.issues()).isEmpty();
        var run = canvasRuns.start(f.canvas(), new CanvasRunData.Start("business-creation", plan.id(), plan.hash(), new GenerationClientScope(f.user(), null)),
                ResourceCommandContext.user(actor(f), f.canvas()));
        String nodeRun = run.nodes().stream().filter(node -> node.nodeId().equals(ids.path("$create").asText())).findFirst().orElseThrow().id();
        await().atMost(Duration.ofSeconds(30)).pollInterval(Duration.ofMillis(200)).untilAsserted(() -> {
            var current = canvasRuns.view(run.id(), actor(f));
            assertThat(current.status()).withFailMessage("正式内容创作：%s", current).isEqualTo(CanvasRunData.State.SUCCEEDED);
            var result = current.nodes().stream().filter(node -> node.id().equals(nodeRun)).findFirst().orElseThrow().outputs().at("results");
            assertThat(result).hasSize(count).allSatisfy(value -> {
                assertThat(value.resource().type()).isEqualTo(CanvasValueType.fromCode(resultType));
                assertThat(value.resource().id()).isNotEqualTo(f.project());
            });
        });
        var task = agentDetails.detail(run.id(), nodeRun, actor(f)).tasks().getFirst();
        assertAutomaticNodeTasks(nodeRun);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command WHERE agent_run_id=? AND tool_call_id='business-create'", Long.class, task.id())).isEqualTo(1);
        if (asset) assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset WHERE project_id=? AND name='雨夜灯塔' AND deleted=FALSE", Long.class, f.project())).isEqualTo(1);
        else assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_storyboard_item WHERE storyboard_id=? AND deleted=FALSE", Long.class, f.project())).isEqualTo(3);
        assertThat(calls).hasValue(2);
    }

    @Test void professionalTaskCannotEscapeItsNodeBudgetThroughConversationContinuation() throws Exception {
        var f = professional(fixture());
        assertThatThrownBy(() -> scopes.resume(mapper.writeValueAsString(f.scope()), f.project(), actor(f)))
                .hasMessageContaining("重新计划运行");
    }

    @Test void canvasLibraryFindsUnplacedResourcesAndRejectsOtherOwnedProjects() throws Exception {
        var f = fixture();
        var library = mapper.readTree(tools.findExecutor("query_canvas_library").execute("{\"category\":\"project\"}", invocation(f, "library")));
        assertThat(library.path("status").asText()).isEqualTo("success");
        assertThat(library.path("items").toString()).contains("script", String.valueOf(f.project()));
        assertThat(canvases.snapshot(f.canvas(), actor(f)).graph().nodes()).isEmpty();
        long other = IDS.incrementAndGet();
        jdbc.update("INSERT INTO afv_project(id,name,owner_type,owner_id) VALUES(?,'同用户其他项目',1,?)", other, f.user());
        jdbc.update("INSERT INTO afv_script(id,project_id,title) VALUES(?,?,'其他项目剧本')", other, other);
        var denied = mapper.readTree(tools.findExecutor("get_canvas_library_resource").execute("{\"type\":\"script\",\"id\":" + other + "}", invocation(f, "library-denied")));
        assertThat(denied.path("code").asInt()).isEqualTo(403);
        assertThat(denied.toString()).doesNotContain("其他项目剧本");
    }

    @Test void nodePromptIsPagedInsteadOfCopyingTheWholeConfigurationIntoEveryRead() throws Exception {
        var f = fixture();
        String content = "正文片段".repeat(10000);
        var input = nodes("$large", "长正文");
        ((ObjectNode) input.get(0).path("node").path("config")).put("content", content);
        String nodeId = mapper.valueToTree(mutations.apply(input, invocation(f, "large-node"))).path("idMappings").path("$large").asText();
        var result = mapper.readTree(tools.findExecutor("get_canvas_node_details").execute(
                "{\"nodeId\":\"" + nodeId + "\",\"offset\":0,\"limit\":512}", invocation(f, "large-read")));
        assertThat(result.path("configuration").path("text").asText()).hasSize(512);
        assertThat(result.path("configuration").path("hasMore").asBoolean()).isTrue();
        assertThat(result.path("node").has("config")).isFalse();
        assertThat(result.toString().length()).isLessThan(6000);
    }

    private AiModel model(Fixture f) {
        when(providers.agentScopeModelFingerprint(any(AiModel.class))).thenAnswer(call -> AgentKernelKey.modelRecordFingerprint(call.getArgument(0)));
        jdbc.update("INSERT INTO afv_api_config(id,name,platform,text_protocol,api_url,api_key) VALUES(?,'调用账本测试','openai_compatible','openai_compatible','http://127.0.0.1:1','test-key')", f.project());
        jdbc.update("""
                INSERT INTO afv_ai_model(id,name,code,model_type,model_protocol,api_config_id,config,max_concurrency,multimodal_input_types,multimodal_input_transports,reasoning_effort_levels)
                VALUES(?,'调用账本测试','test-ledger',1,'openai_compatible',?,'{}',1,JSON_ARRAY(),JSON_OBJECT(),JSON_ARRAY())
                """, f.project(), f.project());
        return modelService.getById(f.project());
    }
    private RuntimeContext runtime(Fixture f) {
        return RuntimeContext.builder().userId(String.valueOf(f.user())).sessionId(f.runId())
                .put(AuthenticatedUserContext.class, new AuthenticatedUserContext(f.user()))
                .put(AgentRunContext.class, new AgentRunContext(f.runId(), "agent-test", 1L, Instant.now().plusSeconds(3600))).build();
    }

    @Test void professionalScopeCannotWriteOtherObjectsOrReadAnotherProjectEvenIfOwned() throws Exception {
        var f = professional(fixture());
        var other = fixture();
        jdbc.update("INSERT INTO afv_project_member(project_id,user_id,role) VALUES(?,?,1)", other.project(), f.user());
        assertThatThrownBy(() -> resources.patch(new ResourcePatchCommand("scope-project", CanvasValueType.PROJECT, f.project(), 0,
                mapper.valueToTree(Map.of("name", "越界修改"))), invocation(f, "project").requireMutationContext())).hasMessageContaining("输入范围");
        assertThatThrownBy(() -> nodeTools.before("get_project", Map.of("projectId", other.project()), invocation(f, "cross-project")))
                .hasMessageContaining("项目范围");
        assertThatThrownBy(() -> nodeTools.before("get_script_episode", Map.of("scriptId", other.project()), invocation(f, "cross-resource")))
                .hasMessageContaining("当前画布项目");
        assertThatThrownBy(() -> nodeTools.before("list_project_assets", Map.of(), invocation(f, "all-assets")))
                .hasMessageContaining("projectId");
    }

    @Test void professionalPreprocessorUsesItsDeclaredCsvEpisodeSchemaWithoutCrossingProjectScope() throws Exception {
        var f = professional(fixture());
        long episode = IDS.incrementAndGet(), otherProject = IDS.incrementAndGet(), otherEpisode = IDS.incrementAndGet();
        jdbc.update("INSERT INTO afv_script_episode(id,script_id,episode_number,title) VALUES(?,?,1,'当前分集')", episode, f.project());
        jdbc.update("INSERT INTO afv_project(id,name,owner_type,owner_id) VALUES(?,'其他项目',1,?)", otherProject, f.user());
        jdbc.update("INSERT INTO afv_script(id,project_id,title) VALUES(?,?,'其他剧本')", otherProject, otherProject);
        jdbc.update("INSERT INTO afv_script_episode(id,script_id,episode_number,title) VALUES(?,?,1,'其他分集')", otherEpisode, otherProject);
        nodeTools.before("storyboard_asset_preprocessor", Map.of("scriptEpisodeIds", String.valueOf(episode)), invocation(f, "csv-scope"));
        assertThatThrownBy(() -> nodeTools.before("storyboard_asset_preprocessor", Map.of("scriptEpisodeIds", episode + "," + otherEpisode), invocation(f, "csv-other")))
                .hasMessageContaining("当前画布项目");
        assertThatThrownBy(() -> nodeTools.before("storyboard_asset_preprocessor", Map.of("scriptEpisodeIds", episode + ",oops"), invocation(f, "csv-invalid")))
                .hasMessageContaining("正整数");
    }

    private Fixture professional(Fixture f) throws Exception {
        String nodeRun = UUID.randomUUID().toString();
        var canvasScope = new CanvasAgentScope(f.canvas(), f.project(), List.of(), List.of(), null, 1, 1, 1, nodeRun);
        var scope = new CanvasAgentNodeScope("script_full_parse", Map.of(CanvasAgentNodeScope.key(CanvasValueType.SCRIPT, f.project()),
                new CanvasAgentNodeScope.Entry(CanvasValueType.SCRIPT, f.project(), 0, true, false, false)),
                Set.of(CanvasValueType.SCRIPT_EPISODE, CanvasValueType.SCRIPT_SCENE), null, null, 0, 0, 16);
        jdbc.update("UPDATE afv_agent_run SET canvas_scope_json=? WHERE run_id=?", mapper.writeValueAsString(canvasScope), f.runId());
        jdbc.update("""
                INSERT INTO afv_canvas_agent_node_job(node_run_id,canvas_run_id,agent_run_id,conversation_id,snapshot_json,input_json,
                    scope_json,dispatch_state,created_at,updated_at) VALUES(?,?,?,?, '{}','{}',?,'DISPATCHED',UTC_TIMESTAMP(3),UTC_TIMESTAMP(3))
                """, nodeRun, UUID.randomUUID().toString(), f.runId(), f.runId(), mapper.writeValueAsString(scope));
        return new Fixture(f.user(), f.project(), f.canvas(), canvasScope, f.runId());
    }

    private void assertAutomaticNodeTasks(String nodeRunId) throws Exception {
        var snapshots = jdbc.queryForList("SELECT agent_definition_snapshot_json FROM afv_agent_run WHERE JSON_UNQUOTE(JSON_EXTRACT(canvas_scope_json,'$.nodeRunId'))=?", String.class, nodeRunId);
        assertThat(snapshots).isNotEmpty();
        for (String snapshot : snapshots)
            assertThat(mapper.readTree(snapshot).path("promptVariables").path("agentToolExecutionMode").asText()).isEqualTo("FULL_ACCESS");
    }

    private Fixture fixture() throws Exception {
        // 为真实命令自动创建的子对象保留区间，避免后续测试的显式 ID 撞上 AUTO_INCREMENT。
        long id = IDS.addAndGet(10_000);
        jdbc.update("INSERT INTO sys_user(id, username, password) VALUES (?, ?, 'test')", id, "canvas-agent-" + id);
        jdbc.update("INSERT INTO afv_project(id, name, owner_type, owner_id) VALUES (?, '助手验收项目', 1, ?)", id, id);
        jdbc.update("INSERT INTO afv_script(id, project_id, title) VALUES (?, ?, '剧本')", id, id);
        String canvas = canvas(id, id);
        var scope = scopes.capture(new CanvasAgentContext(canvas, List.of(), List.of(), null, 1L), id, new CanvasActor(id, null));
        String run = UUID.randomUUID().toString();
        jdbc.update("""
                INSERT INTO afv_agent_run(run_id, conversation_id, user_id, project_id, canvas_scope_json, kernel_fingerprint,
                    agent_definition_snapshot_json, agent_state_session_id, status, owner_instance_id, owner_epoch, lease_until, deadline_at, started_at)
                VALUES (?, ?, ?, ?, ?, 'test', '{}', ?, 'RUNNING', 'agent-test', 1,
                    DATE_ADD(UTC_TIMESTAMP(3), INTERVAL 10 MINUTE), DATE_ADD(UTC_TIMESTAMP(3), INTERVAL 1 HOUR), UTC_TIMESTAMP(3))
                """, run, run, id, id, mapper.writeValueAsString(scope), run);
        return new Fixture(id, id, canvas, scope, run);
    }
    private String canvas(long user, long project) {
        var headers = new HttpHeaders(); headers.setBearerAuth(tokens.createToken(user, "canvas-agent-" + user, null).getAccessToken()); headers.setContentType(MediaType.APPLICATION_JSON);
        String id = UUID.randomUUID().toString();
        var result = http.postForEntity("/api/canvases", new HttpEntity<>(Map.of("id", id, "title", "助手画布", "projectId", project), headers), JsonNode.class);
        assertThat(result.getStatusCode()).isEqualTo(HttpStatus.OK); assertThat(result.getBody().path("code").asInt()).isZero();
        return id;
    }
    private CanvasActor actor(Fixture f) { return new CanvasActor(f.user(), null); }
    private ToolExecutionContext invocation(Fixture f, String call) {
        var context = ToolExecutionContext.forInvocation(f.user(), 1, f.user(), new ResourceCommandContext(actor(f), "agent", f.canvas(), f.runId(), call, "agent-test", 1L));
        context.setCanvasScope(f.scope()); return context;
    }
    private ArrayNode nodes(String alias, String title) throws Exception {
        return (ArrayNode) mapper.readTree(mapper.writeValueAsString(List.of(Map.of("type", "canvas.node.create", "node", Map.of(
                "id", alias, "kind", "text", "schemaVersion", 1, "title", title, "config", Map.of("content", "正文"), "configVersion", 0,
                "layoutVersion", 0, "layout", Map.of("x", 40, "y", 80, "width", 300, "locked", false, "collapsed", false))))));
    }
}
