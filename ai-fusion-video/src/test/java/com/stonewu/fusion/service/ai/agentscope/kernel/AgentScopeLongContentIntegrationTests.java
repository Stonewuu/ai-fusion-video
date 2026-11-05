package com.stonewu.fusion.service.ai.agentscope.kernel;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.config.AgentScopeRuntimeProperties;
import com.stonewu.fusion.config.AgentScopeV2Properties;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.ToolExecutor;
import com.stonewu.fusion.service.ai.agentscope.AgentScopeToolAdapter;
import com.stonewu.fusion.service.ai.agentscope.context.AgentRunContext;
import com.stonewu.fusion.service.ai.agentscope.context.CancellationContext;
import com.stonewu.fusion.service.ai.agentscope.context.ToolExecutionContext;
import com.stonewu.fusion.service.ai.agentscope.runtime.AgentRuntimeSchedulers;
import com.stonewu.fusion.service.ai.agentscope.skill.AgentScopeSkillRegistry;
import com.stonewu.fusion.service.ai.agentscope.state.AgentScopeShutdownRecoveryBridge;
import com.stonewu.fusion.service.ai.agentscope.state.InMemoryStateStoreFailureGuard;
import com.stonewu.fusion.service.ai.agentscope.tool.AgentScopeToolSchema;
import com.stonewu.fusion.service.ai.agentscope.workspace.AgentWorkspaceBaseStore;
import com.stonewu.fusion.service.ai.run.RunLeaseGuard;
import com.stonewu.fusion.service.ai.text.LongToolResultStore;
import com.stonewu.fusion.service.ai.tool.ReadToolResultToolExecutor;
import com.stonewu.fusion.service.canvas.CanvasJson;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.message.ContentBlock;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.state.InMemoryAgentStateStore;
import io.agentscope.harness.agent.filesystem.remote.store.StoreItem;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class AgentScopeLongContentIntegrationTests {
    @Test
    @DisplayName("运行完整 Harness 工具循环，模型按可用工具连续读完长结果，不再调用 read_file")
    void harnessReadsAllPagesThroughTheRegisteredPlatformAdapter() {
        var json = new ObjectMapper();
        var workspace = mock(AgentWorkspaceBaseStore.class);
        Map<String, StoreItem> stored = new ConcurrentHashMap<>();
        when(workspace.get(any(), anyString())).thenAnswer(call -> stored.get(call.getArgument(1)));
        when(workspace.putArtifactIfAbsent(any(), anyString(), any())).thenAnswer(call -> stored.putIfAbsent(
                call.getArgument(1), new StoreItem(call.getArgument(1), call.getArgument(2), 1)) == null);
        var results = new LongToolResultStore(workspace);
        var reader = new ReadToolResultToolExecutor(results, new CanvasJson(json));
        String fullText = "开头" + "中".repeat(95_906) + "结尾";
        ToolExecutor large = mock(ToolExecutor.class);
        when(large.getToolName()).thenReturn("load_large_result");
        when(large.getToolDescription()).thenReturn("读取长文本测试结果");
        when(large.getParametersSchema()).thenReturn("{\"type\":\"object\",\"properties\":{}}");
        when(large.isReadOnly()).thenReturn(true);
        when(large.isConcurrencySafe()).thenReturn(true);
        when(large.execute(anyString(), any())).thenReturn(fullText);
        List<ToolExecutor> executors = List.of(large, reader);
        AiModel record = AiModel.builder().id(7L).code("paging-test").contextWindow(200_000).build();
        var manifest = executors.stream().map(tool -> new AgentKernelToolManifest(tool.getToolName(),
                AgentKernelToolManifest.schemaSha256(AgentScopeToolSchema.prepare(json, tool.getParametersSchema(),
                        tool.getToolName()).canonicalJson()), true, true)).toList();
        var key = AgentKernelKey.create("paging", "model", AgentKernelKey.promptVersion("分段读取全文"),
                manifest, AgentKernelSpecFactory.TOOL_WHITELIST_VERSION);
        var spec = new AgentKernelSpec(key, record, "paging", "分段读取测试", "检查长内容工具链", "分段读取全文",
                30, manifest, Set.of("load_large_result", "read_tool_result"), AgentKernelSpecFactory.TOOL_WHITELIST_VERSION);
        var lease = mock(RunLeaseGuard.class);
        when(lease.assertLease(anyString(), anyString(), anyLong())).thenReturn(Mono.empty());
        var state = new InMemoryAgentStateStore();
        var model = new PagingModel(json);
        try (var schedulers = new AgentRuntimeSchedulers(new AgentScopeRuntimeProperties())) {
            var factory = new AgentScopeHarnessFactory(ignored -> OwnedChatModel.owned(model), (ignored, toolkit) -> {
                for (ToolExecutor executor : executors) {
                    toolkit.registerAgentTool(new AgentScopeToolAdapter(executor, AgentScopeToolSchema.prepare(json,
                            executor.getParametersSchema(), executor.getToolName()), schedulers.toolBlocking(), lease,
                            json, record, List.of()));
                }
                return AgentKernelToolkitResources.none();
            }, state, new InMemoryStateStoreFailureGuard(), new AgentScopeShutdownRecoveryBridge(state, schedulers),
                    new AgentScopeSkillRegistry(new AgentScopeV2Properties()), null, results);
            var context = RuntimeContext.builder().userId("7").sessionId("paging-session")
                    .put(AgentRunContext.class, new AgentRunContext("run", "worker", 1L, Instant.now().plusSeconds(60)))
                    .put(CancellationContext.class, CancellationContext.noop())
                    .put(ToolExecutionContext.class, new ToolExecutionContext(7L, 1, 7L, null)).build();
            try (var resource = factory.create(spec)) {
                assertThat(resource.agent().getToolkit().getToolNames())
                        .containsExactlyInAnyOrder("load_large_result", "read_tool_result");
                var reply = resource.agent().call(List.of(Msg.builder().role(MsgRole.USER)
                        .content(TextBlock.builder().text("读取完整结果").build()).build()), context)
                        .block(Duration.ofSeconds(30));
                assertThat(reply.getTextContent()).isEqualTo("全文读取完成");
            }
        }
        assertThat(model.read.toString()).isEqualTo(fullText);
        verify(large, times(1)).execute(anyString(), any());
    }

    private static final class PagingModel extends ChatModelBase {
        private final ObjectMapper json;
        private final StringBuilder read = new StringBuilder();
        private int calls;

        private PagingModel(ObjectMapper json) {
            this.json = json;
        }

        @Override
        protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> tools, GenerateOptions options) {
            try {
                calls++;
                ContentBlock output;
                String finish = "tool_calls";
                if (calls == 1) {
                    output = call("load_large_result", Map.of());
                } else {
                    var result = messages.stream().flatMap(message -> message.getContent().stream())
                            .filter(ToolResultBlock.class::isInstance).map(ToolResultBlock.class::cast)
                            .reduce((previous, next) -> next).orElseThrow();
                    String text = ((TextBlock) result.getOutput().getFirst()).getText();
                    assertThat(text).doesNotStartWith("Error:").doesNotContain("Tool not found", "read_file");
                    JsonNode body = json.readTree(text);
                    int offset = 0;
                    if (result.getName().equals("read_tool_result")) {
                        assertThat(body.path("status").asText()).isEqualTo("success");
                        assertThat(body.path("offset").asInt()).isEqualTo(read.length());
                        read.append(body.path("content").asText());
                        offset = body.path("nextOffset").asInt();
                    }
                    if (result.getName().equals("read_tool_result") && !body.path("hasMore").asBoolean()) {
                        output = TextBlock.builder().text("全文读取完成").build();
                        finish = "stop";
                    } else {
                        output = call("read_tool_result", Map.of("resultId", body.path("resultId").asText(),
                                "contentVersion", body.path("contentVersion").asText(), "offset", offset, "limit", 8000));
                    }
                }
                return Flux.just(ChatResponse.builder().id("reply-" + calls).content(List.of(output)).build(),
                        ChatResponse.builder().id("reply-" + calls).content(List.of()).finishReason(finish).build());
            } catch (Exception failure) {
                return Flux.error(failure);
            }
        }

        private ToolUseBlock call(String name, Map<String, Object> input) throws JsonProcessingException {
            return ToolUseBlock.builder().id("call-" + calls).name(name).input(input)
                    .content(json.writeValueAsString(input)).build();
        }

        @Override
        public String getModelName() {
            return "paging-test";
        }
    }
}
