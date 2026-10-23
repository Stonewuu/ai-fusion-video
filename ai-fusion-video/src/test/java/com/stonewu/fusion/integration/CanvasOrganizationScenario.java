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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.assertj.core.api.Assertions.assertThat;

/** 受控模型通过实际 SDK 工具逐层发现第二集，所有创建引用均来自工具返回的 ID/版本。 */
final class CanvasOrganizationScenario extends ChatModelBase {
    private final ObjectMapper mapper;
    private final CanvasJson json;
    private final Map<String, JsonNode> found = new HashMap<>();
    private final AtomicInteger steps = new AtomicInteger();
    private JsonNode created;
    private volatile JsonNode lastResult;
    private String videoId;

    CanvasOrganizationScenario(ObjectMapper mapper) { this.mapper = mapper; this.json = new CanvasJson(mapper); }
    @Override public String getModelName() { return "organize-existing-episode-scenario"; }
    int calls() { return steps.get(); }
    JsonNode lastResult() { return lastResult; }
    JsonNode created() { return created; }
    private record Call(String name, Map<String, Object> arguments) { }

    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
        int step = steps.getAndIncrement();
        Call next = switch (step) {
            case 0 -> new Call("get_canvas_context", Map.of("kinds", List.of("group", "scriptEpisode", "scriptScene", "asset", "storyboardEpisode", "shot", "video", "note")));
            case 1 -> new Call("query_canvas_library", Map.of("category", "project"));
            case 2 -> {
                for (var item : last(messages, "query_canvas_library").path("items")) found.put(item.path("nodeKind").asText(), item);
                yield children("script");
            }
            case 3 -> {
                found.put("episode", named(last(messages, "query_canvas_library"), "第二集 · 雨夜灯塔"));
                yield children("episode");
            }
            case 4 -> {
                var rows = last(messages, "query_canvas_library").path("items");
                assertThat(rows).hasSize(1);
                found.put("scene", rows.get(0));
                yield details("scene");
            }
            case 5 -> {
                var scene = body(messages);
                assertThat(scene.path("characterAssetIds")).hasSize(1);
                yield new Call("get_canvas_library_resource", Map.of("type", "asset", "id", scene.path("characterAssetIds").get(0).asLong()));
            }
            case 6 -> {
                var asset = body(messages);
                found.put("asset", mapper.valueToTree(Map.of("nodeKind", "asset", "resource", Map.of(
                        "type", "asset", "id", asset.path("id").asLong(), "version", asset.path("version").asInt()))));
                yield children("storyboard");
            }
            case 7 -> {
                found.put("boardEpisode", named(last(messages, "query_canvas_library"), "第二集 · 雨夜灯塔"));
                yield details("boardEpisode");
            }
            case 8 -> {
                assertThat(body(messages).path("scriptEpisodeId").asLong()).isEqualTo(resourceId("episode"));
                yield children("boardEpisode");
            }
            case 9 -> {
                var rows = last(messages, "query_canvas_library").path("items");
                assertThat(rows).hasSize(1);
                found.put("boardScene", rows.get(0));
                yield children("boardScene");
            }
            case 10 -> {
                var rows = last(messages, "query_canvas_library").path("items");
                assertThat(rows).hasSize(2);
                found.put("missing", rows.get(0)); found.put("ready", rows.get(1));
                yield details("missing");
            }
            case 11 -> {
                var shot = body(messages);
                assertThat(shot.hasNonNull("firstFrameImageUrl")).isFalse();
                String video = shot.path("videoUrl").asText();
                assertThat(video).startsWith("media://");
                videoId = video.substring("media://".length());
                yield details("ready");
            }
            case 12 -> {
                assertThat(body(messages).path("firstFrameImageUrl").asText()).startsWith("media://");
                yield new Call("query_canvas_library", Map.of("category", "media"));
            }
            case 13 -> {
                for (var item : last(messages, "query_canvas_library").path("items")) {
                    if (item.path("mediaId").asText().equals(videoId)) found.put("video", item);
                }
                assertThat(found).containsKey("video");
                yield new Call("apply_canvas_operations", Map.of("operations", operations()));
            }
            case 14 -> {
                created = last(messages, "apply_canvas_operations");
                assertThat(created.path("status").asText()).as(created.toString()).isEqualTo("success");
                var ids = List.of("episode", "scene", "asset", "boardEpisode", "missing", "ready", "video", "note").stream()
                        .map(key -> created.path("idMappings").path("$" + key).asText()).toList();
                yield new Call("arrange_canvas_nodes", Map.of("nodeIds", ids, "mode", "arrange", "expectedRevision", created.path("canvasRevision").asLong()));
            }
            case 15 -> {
                assertThat(last(messages, "arrange_canvas_nodes").path("status").asText()).isEqualTo("success");
                yield new Call("focus_canvas_nodes", Map.of("nodeIds", List.of(created.path("idMappings").path("$group").asText())));
            }
            case 16 -> {
                assertThat(last(messages, "focus_canvas_nodes").path("status").asText()).isEqualTo("success");
                yield null;
            }
            default -> throw new AssertionError("整理场景出现了额外模型调用");
        };
        if (next == null) return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder()
                .text("第二集的已有内容已整理，缺首帧镜头已标注；没有生成新素材。").build())).finishReason("stop").build());
        assertThat(schema).anyMatch(tool -> tool.getName().equals(next.name()));
        Map<String, Object> arguments = mapper.convertValue(next.arguments(), Map.class);
        return Flux.just(ChatResponse.builder().content(List.of(ToolUseBlock.builder().id("organize-" + step)
                .name(next.name()).input(arguments).content(json.write(arguments)).build())).finishReason("tool_calls").build());
    }

    private Call children(String key) {
        var item = found.get(key);
        return new Call("query_canvas_library", Map.of("category", "children", "parentType", item.path("nodeKind").asText(), "parentId", resourceId(key)));
    }
    private Call details(String key) {
        return new Call("get_canvas_library_resource", Map.of("type", found.get(key).path("nodeKind").asText(), "id", resourceId(key)));
    }
    private long resourceId(String key) { return found.get(key).at("/resource/id").asLong(); }
    private JsonNode body(List<Msg> messages) { return json.read(last(messages, "get_canvas_library_resource").at("/content/text").asText(), JsonNode.class); }
    private JsonNode named(JsonNode result, String title) {
        for (var item : result.path("items")) if (item.path("title").asText().equals(title)) return item;
        throw new AssertionError("实际目录没有请求的第二集");
    }
    private JsonNode last(List<Msg> messages, String name) {
        var results = messages.stream().flatMap(message -> message.getContent().stream()).filter(ToolResultBlock.class::isInstance)
                .map(ToolResultBlock.class::cast).filter(block -> block.getName().equals(name)).toList();
        assertThat(results).isNotEmpty();
        String text = results.getLast().getOutput().stream().filter(TextBlock.class::isInstance).map(TextBlock.class::cast).map(TextBlock::getText).reduce("", String::concat);
        var result = json.read(text, JsonNode.class);
        lastResult = result;
        assertThat(result.path("status").asText()).as(result.toString()).isEqualTo("success");
        return result;
    }
    private List<Object> operations() {
        List<Object> operations = new ArrayList<>();
        operations.add(create("group", "group", "第二集 · 雨夜灯塔", Map.of(), null, 40, 80));
        String[] keys = {"episode", "scene", "asset", "boardEpisode", "missing", "ready", "video"};
        String[] labels = {"第二集剧本", "灯塔外景", "阿青", "第二集分镜", "镜头 2-1 · 缺少首帧", "镜头 2-2", "已有镜头视频"};
        for (int index = 0; index < keys.length; index++) {
            var item = found.get(keys[index]);
            operations.add(create(keys[index], item.path("nodeKind").asText(), labels[index],
                    item.has("config") ? mapper.convertValue(item.path("config"), Map.class) : Map.of(),
                    item.get("resource"), 40 + (index % 3) * 380, 80 + (index / 3) * 260));
        }
        operations.add(create("note", "note", "缺少首帧", Map.of("content", "镜头 2-1 尚无首帧；已有视频仍可预览，本次未生成素材。"), null, 420, 600));
        operations.add(Map.of("type", "canvas.edge.create", "edge", Map.of("id", "$missing-note", "kind", "annotation",
                "sourceNodeId", "$missing", "targetNodeId", "$note", "version", 0)));
        return operations;
    }
    private Object create(String key, String kind, String title, Map<String, Object> config, JsonNode resource, int x, int y) {
        Map<String, Object> layout = new LinkedHashMap<>(Map.of("x", x, "y", y, "width", 300, "locked", false, "collapsed", false));
        if (key.equals("group")) { layout.put("width", 1440); layout.put("height", 1000); }
        else layout.put("parentId", "$group");
        Map<String, Object> node = new LinkedHashMap<>(Map.of("id", "$" + key, "kind", kind, "title", title, "schemaVersion", 1,
                "config", config, "configVersion", 0, "layoutVersion", 0, "layout", layout));
        if (resource != null) node.put("resource", resource);
        return Map.of("type", "canvas.node.create", "node", node);
    }
}
