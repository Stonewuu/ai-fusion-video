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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

/** 用真实选区、工具返回的版本和素材 ID 完成两镜头修改、生成、旧新对照及批准采用。 */
final class CanvasSelectedShotsScenario extends ChatModelBase {
    private final ObjectMapper mapper;
    private final CanvasJson json;
    private final long videoModel;
    private final AtomicInteger calls = new AtomicInteger();
    private final List<JsonNode> selected = new ArrayList<>();
    private final List<JsonNode> details = new ArrayList<>();
    private final List<JsonNode> output = new ArrayList<>();
    private JsonNode generatedNodes;
    private JsonNode comparisonNodes;
    private JsonNode completed;
    private String characterDescription;
    private volatile String runId;
    private volatile JsonNode lastResult;

    CanvasSelectedShotsScenario(ObjectMapper mapper, long videoModel) {
        this.mapper = mapper; this.json = new CanvasJson(mapper); this.videoModel = videoModel;
    }
    @Override public String getModelName() { return "selected-shots-rain-night"; }
    int calls() { return calls.get(); }
    String runId() { return runId; }
    JsonNode lastResult() { return lastResult; }
    List<JsonNode> outputs() { return List.copyOf(output); }
    private record Call(String name, Map<String, Object> arguments) { }

    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
        int step = calls.getAndIncrement();
        if (step == 18) return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder()
                .text("所选两个镜头已改为雨夜，保留角色造型和旧视频；新视频已生成并经逐项批准采用，其他镜头和分集未修改。").build()))
                .finishReason("stop").build());
        Call next = switch (step) {
            case 0 -> new Call("get_canvas_context", Map.of("kinds", List.of("shot", "videoGeneration", "video")));
            case 1 -> {
                var context = last(messages, "get_canvas_context");
                context.path("selectedNodes").forEach(selected::add);
                assertThat(selected).hasSize(2);
                assertThat(context.at("/scope/selectedNodeIds")).hasSize(2);
                yield detail(0);
            }
            case 2 -> { details.add(last(messages, "get_canvas_node_details")); yield detail(1); }
            case 3 -> {
                details.add(last(messages, "get_canvas_node_details"));
                yield new Call("get_canvas_library_resource", Map.of("type", "assetItem", "id", body(details.get(0)).path("characterIds").get(0).asLong()));
            }
            case 4 -> {
                var character = body(last(messages, "get_canvas_library_resource"));
                characterDescription = character.at("/properties/description").asText();
                assertThat(characterDescription).contains("蓝色雨衣");
                yield new Call("get_canvas_library_resource", Map.of("type", "assetItem", "id", body(details.get(0)).path("sceneAssetItemId").asLong()));
            }
            case 5 -> {
                assertThat(body(last(messages, "get_canvas_library_resource")).path("name").asText()).contains("灯塔");
                yield new Call("get_canvas_models", Map.of("modelId", videoModel));
            }
            case 6 -> {
                assertThat(last(messages, "get_canvas_models").at("/model/inputs/firstFrame").isObject()).isTrue();
                yield patch(0);
            }
            case 7 -> { assertSuccess(messages, "edit_canvas_resource"); yield patch(1); }
            case 8 -> {
                assertSuccess(messages, "edit_canvas_resource");
                var operations = new ArrayList<Object>();
                for (int index = 0; index < 2; index++) {
                    operations.add(node("$generate" + index, "videoGeneration", "雨夜视频 " + index,
                            Map.of("modelId", videoModel, "duration", 4, "count", 1, "resolution", "1280x720", "ratio", "16:9"), 440, index * 360));
                    operations.add(node("$old" + index, "video", "修改前视频 " + index,
                            Map.of("resourceId", body(details.get(index)).path("videoUrl").asText().substring("media://".length())), 840, index * 360));
                    operations.add(edge("$prompt" + index, selected.get(index).path("id").asText(), "prompt", "$generate" + index, "prompt"));
                    operations.add(edge("$frame" + index, selected.get(index).path("id").asText(), "firstFrame", "$generate" + index, "firstFrame"));
                }
                yield new Call("apply_canvas_operations", Map.of("operations", operations));
            }
            case 9 -> {
                generatedNodes = last(messages, "apply_canvas_operations");
                assertThat(generatedNodes.path("status").asText()).as(generatedNodes.toString()).isEqualTo("success");
                yield new Call("validate_canvas_run", Map.of("expectedRevision", generatedNodes.path("canvasRevision").asLong(),
                        "mode", "DEPENDENCIES", "nodeIds", List.of(generatedId(0), generatedId(1)), "reuse", Map.of(),
                        "budget", Map.of("maxNodes", 4, "maxImages", 0, "maxVideoSeconds", 8, "maxParallel", 2, "maxBatchItems", 2)));
            }
            case 10 -> {
                var plan = last(messages, "validate_canvas_run");
                assertThat(plan.path("status").asText()).as(plan.toString()).isEqualTo("ready");
                yield new Call("run_canvas_nodes", Map.of("planId", plan.at("/plan/id").asText(), "planHash", plan.at("/plan/hash").asText()));
            }
            case 11 -> {
                runId = last(messages, "run_canvas_nodes").path("canvasRunId").asText();
                yield new Call("await_canvas_run", Map.of("canvasRunId", runId));
            }
            case 12 -> {
                completed = last(messages, "await_canvas_run");
                assertThat(completed.path("status").asText()).as(completed.toString()).isEqualTo("SUCCEEDED");
                assertThat(completed.path("nodes")).hasSize(4);
                yield outputs(0);
            }
            case 13 -> { output.add(firstOutput(messages)); yield outputs(1); }
            case 14 -> {
                output.add(firstOutput(messages));
                yield new Call("apply_canvas_operations", Map.of("operations", List.of(
                        node("$new0", "video", "雨夜新视频 0", Map.of("resourceId", output.get(0).path("mediaId").asText()), 1220, 0),
                        node("$new1", "video", "雨夜新视频 1", Map.of("resourceId", output.get(1).path("mediaId").asText()), 1220, 360))));
            }
            case 15 -> {
                comparisonNodes = last(messages, "apply_canvas_operations");
                assertThat(comparisonNodes.path("status").asText()).isEqualTo("success");
                yield bind(0);
            }
            case 16 -> { assertSuccess(messages, "bind_canvas_output"); yield bind(1); }
            case 17 -> {
                assertSuccess(messages, "bind_canvas_output");
                yield new Call("focus_canvas_nodes", Map.of("nodeIds", List.of(
                        generatedNodes.at("/idMappings/$old0").asText(), comparisonNodes.at("/idMappings/$new0").asText(),
                        generatedNodes.at("/idMappings/$old1").asText(), comparisonNodes.at("/idMappings/$new1").asText())));
            }
            default -> throw new AssertionError("意外模型调用：" + step);
        };
        assertThat(schema).anyMatch(tool -> tool.getName().equals(next.name()));
        return Flux.just(ChatResponse.builder().content(List.of(ToolUseBlock.builder().id("selected-step-" + step)
                .name(next.name()).input(next.arguments()).content(json.write(next.arguments())).build())).finishReason("tool_calls").build());
    }
    private Call detail(int index) {
        return new Call("get_canvas_node_details", Map.of("nodeId", selected.get(index).path("id").asText()));
    }
    private Call patch(int index) {
        return new Call("edit_canvas_resource", Map.of("action", "patch", "nodeId", selected.get(index).path("id").asText(),
                "expectedVersion", details.get(index).at("/resource/version").asInt(), "fields", Map.of(
                        "content", "雨夜，阿青走近灯塔 · 镜头 " + index,
                        "videoPrompt", "雨夜，阿青走近灯塔，保持" + characterDescription + "，镜头 " + index)));
    }
    private Call outputs(int index) {
        for (var node : completed.path("nodes")) if (node.path("nodeId").asText().equals(generatedId(index)))
            return new Call("query_canvas_outputs", Map.of("canvasRunId", runId, "nodeRunId", node.path("nodeRunId").asText(), "port", "videos"));
        throw new AssertionError("缺少已完成视频节点");
    }
    private Call bind(int index) {
        return new Call("bind_canvas_output", Map.of("canvasRunId", runId, "outputId", output.get(index).path("id").asText(),
                "targetNodeId", selected.get(index).path("id").asText(), "binding", "shot:videoUrl", "expectedVersion", 1));
    }
    private String generatedId(int index) { return generatedNodes.path("idMappings").path("$generate" + index).asText(); }
    private JsonNode firstOutput(List<Msg> messages) {
        var items = last(messages, "query_canvas_outputs").path("items");
        assertThat(items).hasSize(1); return items.get(0);
    }
    private void assertSuccess(List<Msg> messages, String tool) {
        var result = last(messages, tool);
        assertThat(result.path("status").asText()).as(result.toString()).isEqualTo("success");
    }
    private JsonNode body(JsonNode detail) { return json.read(detail.at("/content/text").asText(), JsonNode.class); }
    private JsonNode last(List<Msg> messages, String name) {
        var results = messages.stream().flatMap(message -> message.getContent().stream()).filter(ToolResultBlock.class::isInstance)
                .map(ToolResultBlock.class::cast).filter(block -> block.getName().equals(name)).toList();
        assertThat(results).isNotEmpty();
        String content = results.getLast().getOutput().stream().filter(TextBlock.class::isInstance).map(TextBlock.class::cast)
                .map(TextBlock::getText).reduce("", String::concat);
        lastResult = json.read(content, JsonNode.class); return lastResult;
    }
    private Object node(String id, String kind, String title, Map<String, Object> config, int x, int y) {
        return Map.of("type", "canvas.node.create", "node", Map.of("id", id, "kind", kind, "title", title, "schemaVersion", 1,
                "config", config, "configVersion", 0, "layoutVersion", 0, "layout", Map.of("x", x, "y", y, "width", 300, "locked", false, "collapsed", false)));
    }
    private Object edge(String id, String source, String sourcePort, String target, String targetPort) {
        return Map.of("type", "canvas.edge.create", "edge", Map.of("id", id, "kind", "execution", "sourceNodeId", source,
                "sourcePort", sourcePort, "targetNodeId", target, "targetPort", targetPort, "inputIndex", 0, "version", 0));
    }
}
