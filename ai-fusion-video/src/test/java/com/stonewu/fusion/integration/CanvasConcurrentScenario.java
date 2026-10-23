package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.service.canvas.CanvasJson;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.message.ToolResultBlock;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

/** 生成期间由测试中的两个独立用户提交业务编辑，模型只通过正式工具恢复和读取结果。 */
final class CanvasConcurrentScenario extends ChatModelBase {
    private final ObjectMapper mapper;
    private final CanvasJson json;
    private final JsonNode operations;
    private final AtomicInteger calls = new AtomicInteger();
    private volatile JsonNode created;
    private volatile JsonNode lastResult;
    private volatile String runId;

    CanvasConcurrentScenario(ObjectMapper mapper, JsonNode operations) {
        this.mapper = mapper;
        this.json = new CanvasJson(mapper);
        this.operations = operations;
    }
    @Override public String getModelName() { return "concurrent-canvas-scenario"; }
    int calls() { return calls.get(); }
    String runId() { return runId; }
    JsonNode created() { return created; }
    JsonNode lastResult() { return lastResult; }
    private record Call(String name, Map<String, Object> args) { }

    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
        int step = calls.getAndIncrement();
        if (step == 6) {
            assertThat(last(messages, "query_canvas_outputs").path("items")).hasSize(1);
            return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder()
                    .text("参考图已按提交时快照生成；保留两位用户的最新编辑，结果待更新，未自动覆盖业务内容。").build()))
                    .finishReason("stop").build());
        }
        Call next = switch (step) {
            case 0 -> new Call("get_canvas_context", Map.of("kinds", List.of("scriptScene", "shot", "imageGeneration")));
            case 1 -> new Call("apply_canvas_operations", Map.of("operations", mapper.convertValue(operations, List.class)));
            case 2 -> {
                created = last(messages, "apply_canvas_operations");
                assertThat(created.path("status").asText()).withFailMessage("图命令：%s", created).isEqualTo("success");
                yield new Call("validate_canvas_run", Map.of("expectedRevision", created.path("canvasRevision").asLong(),
                        "mode", "DEPENDENCIES", "nodeIds", List.of(created.path("idMappings").path("$image").asText()),
                        "reuse", Map.of(), "budget", Map.of("maxNodes", 2, "maxImages", 1, "maxVideoSeconds", 0,
                                "maxParallel", 1, "maxBatchItems", 1)));
            }
            case 3 -> {
                var plan = last(messages, "validate_canvas_run");
                assertThat(plan.path("status").asText()).withFailMessage("计划：%s", plan).isEqualTo("ready");
                yield new Call("run_canvas_nodes", Map.of("planId", plan.at("/plan/id").asText(), "planHash", plan.at("/plan/hash").asText()));
            }
            case 4 -> {
                runId = last(messages, "run_canvas_nodes").path("canvasRunId").asText();
                assertThat(runId).isNotBlank();
                yield new Call("await_canvas_run", Map.of("canvasRunId", runId));
            }
            case 5 -> {
                var result = last(messages, "await_canvas_run");
                assertThat(result.path("status").asText()).withFailMessage("等待：%s", result).isEqualTo("SUCCEEDED");
                String nodeRunId = null;
                for (var node : result.path("nodes"))
                    if (node.path("nodeId").asText().equals(created.path("idMappings").path("$image").asText()))
                        nodeRunId = node.path("nodeRunId").asText();
                assertThat(nodeRunId).isNotBlank();
                yield new Call("query_canvas_outputs", Map.of("canvasRunId", runId, "nodeRunId", nodeRunId, "port", "images"));
            }
            default -> throw new AssertionError("意外的模型调用次数：" + step);
        };
        return Flux.just(ChatResponse.builder().content(List.of(ToolUseBlock.builder().id("concurrent-step-" + step)
                .name(next.name()).input(next.args()).content(json.write(next.args())).build())).finishReason("tool_calls").build());
    }

    private JsonNode last(List<Msg> messages, String name) {
        var results = messages.stream().flatMap(message -> message.getContent().stream())
                .filter(ToolResultBlock.class::isInstance).map(ToolResultBlock.class::cast)
                .filter(block -> block.getName().equals(name)).toList();
        assertThat(results).isNotEmpty();
        String content = results.getLast().getOutput().stream().filter(TextBlock.class::isInstance)
                .map(TextBlock.class::cast).map(TextBlock::getText).reduce("", String::concat);
        lastResult = json.read(content, JsonNode.class);
        return lastResult;
    }
}
