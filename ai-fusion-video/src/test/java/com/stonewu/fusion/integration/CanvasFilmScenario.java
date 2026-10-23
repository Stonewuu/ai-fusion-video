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

/** 故事/子任务、资产及分镜阶段均通过实际 SDK 工具提交，跨阶段只使用真实写入回执。 */
final class CanvasFilmScenario extends ChatModelBase {
    enum Phase { STORY, ASSETS, SHOTS }
    private final CanvasJson json;
    private final long project;
    private final AtomicInteger parentCalls = new AtomicInteger();
    private final AtomicInteger childCalls = new AtomicInteger();
    private final AtomicInteger assetCalls = new AtomicInteger();
    private final AtomicInteger shotCalls = new AtomicInteger();
    private volatile Phase phase = Phase.STORY;
    private volatile JsonNode lastResult;
    private long episode;
    private int episodeVersion;
    private long scene;
    private long characterItem;
    private long sceneItem;
    private long boardEpisode;
    private int boardEpisodeVersion;
    private final List<Long> shots = new ArrayList<>();

    CanvasFilmScenario(ObjectMapper mapper, long project) { this.json = new CanvasJson(mapper); this.project = project; }
    void phase(Phase value) { phase = value; }
    long episode() { return episode; }
    long scene() { return scene; }
    List<Long> shots() { return List.copyOf(shots); }
    JsonNode lastResult() { return lastResult; }
    int calls() { return parentCalls.get() + childCalls.get() + assetCalls.get() + shotCalls.get(); }
    @Override public String getModelName() { return "story-to-film-contract"; }
    private record Call(String id, String tool, Map<String, Object> args) { }

    @Override protected Flux<ChatResponse> doStream(List<Msg> messages, List<ToolSchema> schema, GenerateOptions options) {
        Call next;
        if (phase == Phase.STORY && schema.stream().anyMatch(tool -> tool.getName().equals("episode_script_creator"))) {
            next = switch (parentCalls.getAndIncrement()) {
                case 0 -> new Call("film-save-episode", "save_script_episode", Map.of("scriptId", project, "expectedVersion", 0,
                        "episodeNumber", 1, "title", "雨夜灯塔", "synopsis", "阿青在灯塔找到一封信", "rawContent", "阿青：我会回来。", "sourceType", 2));
                case 1 -> {
                    var saved = success(messages, "save_script_episode");
                    episode = saved.path("scriptEpisodeId").asLong(); episodeVersion = saved.path("episode_version").asInt();
                    assertThat(episode).isPositive();
                    yield new Call("film-create-scenes", "episode_script_creator", Map.of("scriptEpisodeId", episode));
                }
                case 2 -> {
                    assertThat(json.write(messages)).contains("场次已完成");
                    yield null;
                }
                default -> throw new AssertionError("故事父任务重复调用");
            };
        } else if (phase == Phase.STORY) {
            assertThat(schema).anyMatch(tool -> tool.getName().equals("save_script_scene_items"));
            next = switch (childCalls.getAndIncrement()) {
                case 0 -> new Call("film-save-scenes", "save_script_scene_items", Map.of("scriptEpisodeId", episode,
                        "episode_version", episodeVersion, "overwriteMode", false, "scenes", List.of(Map.of(
                                "scene_heading", "外景 灯塔 夜", "location", "灯塔", "time_of_day", "夜", "int_ext", "外景",
                                "characters", List.of("阿青"), "scene_description", "温暖悬疑的雨夜，蓝色雨衣",
                                "dialogues", List.of(Map.of("type", 1, "character_name", "阿青", "content", "我会回来。"))))));
                case 1 -> {
                    var saved = success(messages, "save_script_scene_items");
                    scene = changedIds(saved, "scriptScene").getFirst();
                    yield null;
                }
                default -> throw new AssertionError("场次子任务重复调用");
            };
        } else if (phase == Phase.ASSETS) {
            next = switch (assetCalls.getAndIncrement()) {
                case 0 -> new Call("film-save-assets", "batch_create_assets", Map.of("projectId", project, "reuseExisting", false, "assets", List.of(
                        Map.of("name", "阿青", "type", "character", "initialItem", Map.of("name", "蓝色雨衣造型", "properties", Map.of("description", "短发、蓝色雨衣"))),
                        Map.of("name", "灯塔", "type", "scene", "initialItem", Map.of("name", "雨夜灯塔", "properties", Map.of("description", "温暖悬疑的雨夜灯塔"))))));
                case 1 -> {
                    var saved = success(messages, "batch_create_assets");
                    var items = changedIds(saved, "assetItem");
                    assertThat(items).hasSize(2); characterItem = items.get(0); sceneItem = items.get(1);
                    yield null;
                }
                default -> throw new AssertionError("资产阶段重复调用");
            };
        } else {
            next = switch (shotCalls.getAndIncrement()) {
                case 0 -> new Call("film-save-board-episode", "save_storyboard_episode", Map.of("storyboardId", project,
                        "expectedParentVersion", 0, "scriptEpisodeId", episode, "episodeNumber", 1, "title", "雨夜灯塔分镜"));
                case 1 -> {
                    var saved = success(messages, "save_storyboard_episode");
                    boardEpisode = saved.path("storyboardEpisodeId").asLong(); boardEpisodeVersion = saved.path("version").asInt();
                    assertThat(boardEpisode).isPositive();
                    yield new Call("film-save-shots", "save_storyboard_scene_shots", Map.of("storyboardId", project,
                            "storyboardEpisodeId", boardEpisode, "expectedParentVersion", boardEpisodeVersion,
                            "sceneNumber", "1-1", "sceneHeading", "外景 灯塔 夜", "shots", List.of(shot(0), shot(1))));
                }
                case 2 -> {
                    success(messages, "save_storyboard_scene_shots").path("createdShotIds").forEach(value -> shots.add(value.asLong()));
                    assertThat(shots).hasSize(2); yield null;
                }
                default -> throw new AssertionError("分镜阶段重复调用");
            };
        }
        if (next == null) return Flux.just(ChatResponse.builder().content(List.of(TextBlock.builder()
                .text(phase == Phase.STORY ? "场次已完成；正式剧本已保存" : "正式创作内容已保存").build())).finishReason("stop").build());
        assertThat(schema).anyMatch(tool -> tool.getName().equals(next.tool()));
        return Flux.just(ChatResponse.builder().content(List.of(ToolUseBlock.builder().id(next.id()).name(next.tool())
                .input(next.args()).content(json.write(next.args())).build())).finishReason("tool_calls").build());
    }
    private Map<String, Object> shot(int index) {
        return Map.of("content", "雨夜镜头 " + index, "duration", 4, "cameraMovement", "固定", "videoPrompt", "镜头 " + index + "：雨夜灯塔，蓝色雨衣",
                "characterIds", List.of(characterItem), "sceneAssetItemId", sceneItem);
    }
    private List<Long> changedIds(JsonNode result, String type) {
        var ids = new ArrayList<Long>();
        result.path("changes").forEach(change -> { if (change.path("resourceType").asText().equals(type)) ids.add(change.path("resourceId").asLong()); });
        assertThat(ids).isNotEmpty(); return ids;
    }
    private JsonNode success(List<Msg> messages, String name) {
        var results = messages.stream().flatMap(message -> message.getContent().stream()).filter(ToolResultBlock.class::isInstance)
                .map(ToolResultBlock.class::cast).filter(block -> block.getName().equals(name)).toList();
        assertThat(results).isNotEmpty();
        String content = results.getLast().getOutput().stream().filter(TextBlock.class::isInstance).map(TextBlock.class::cast)
                .map(TextBlock::getText).reduce("", String::concat);
        lastResult = json.read(content, JsonNode.class);
        assertThat(lastResult.path("status").asText()).as(lastResult.toString()).isEqualTo("success");
        return lastResult;
    }
}
