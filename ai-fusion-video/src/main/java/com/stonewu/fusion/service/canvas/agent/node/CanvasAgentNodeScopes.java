package com.stonewu.fusion.service.canvas.agent.node;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasResourceCollectionService;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.resource.AssetResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardResourceQueryService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

import static com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeScope.Entry;
import static com.stonewu.fusion.service.canvas.agent.node.CanvasAgentNodeScope.key;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;

@Component
@RequiredArgsConstructor
public class CanvasAgentNodeScopes {
    private static final Set<String> SCRIPT_ROOT = Set.of("story_to_script");
    private static final Set<String> BOARD_ROOT = Set.of("script_to_storyboard", "episode_storyboard_writer");
    private static final Set<String> ASSET_ROOTS = Set.of("script_to_storyboard", "storyboard_asset_preprocessor");
    private static final Set<String> SCRIPT_READ_ONLY = Set.of("script_to_storyboard", "episode_storyboard_writer",
            "storyboard_asset_preprocessor", "canvas_asset_extractor", "concept_visualizer");
    private final ResourceQueryService resources;
    private final StoryboardResourceQueryService boards;
    private final AssetResourceQueryService assets;
    private final CanvasResourceCollectionService collections;
    private final CanvasJson json;
    private final CanvasAgentBatchVersions batches;

    /**
     * 这些隐含的业务容器也属于执行计划，不能在执行时悄悄改用另一个容器。
     */
    public List<CanvasResourceReference> targets(CanvasDocument document, CanvasNode node, CanvasActor actor) {
        String type = CanvasAgentNodeTypes.type(node);
        if (document.projectId() == null && !"concept_visualizer".equals(type)) {
            throw CanvasRunException.invalid("专业业务智能体需要项目画布");
        }
        List<CanvasResourceReference> result = new ArrayList<>();
        if (SCRIPT_ROOT.contains(type)) {
            add(result, SCRIPT, resources.script(document.projectId(), actor));
        }
        if (BOARD_ROOT.contains(type)) {
            add(result, STORYBOARD, boards.project(document.projectId(), actor));
        }
        if (ASSET_ROOTS.contains(type)) {
            for (var asset : assets.list(document.projectId(), null, null, false, actor)) {
                add(result, ASSET, asset);
            }
        }
        if (result.size() > 2000) {
            throw CanvasRunException.invalid("本次智能体范围超过 2000 个业务对象，请缩小范围");
        }
        return List.copyOf(result);
    }

    private void add(List<CanvasResourceReference> result, CanvasValueType type, ObjectNode data) {
        if (data == null) {
            throw CanvasRunException.invalid("项目缺少 " + type.code() + " 容器，请先初始化项目工作区");
        }
        result.add(new CanvasResourceReference(type, data.path("id").asLong(), data.path("version").asLong()));
    }

    public CanvasAgentNodeScope capture(Run run, NodeRun node) {
        String type = CanvasAgentNodeTypes.type(node.plan().node());
        Map<String, Entry> entries = new LinkedHashMap<>();
        var previous = batches.previous(node);
        for (var value : node.inputs().all()) {
            if (value.resource() != null) {
                var ref = batches.advance(value.resource(), previous);
                boolean writable = !SCRIPT_READ_ONLY.contains(type);
                addTree(run, ref, writable, entries);
            }
        }
        for (JsonNode encoded : node.plan().environment().path("businessTargets")) {
            var target = json.read(encoded.toString(), CanvasResourceReference.class);
            var supplied = node.inputs().all().stream().filter(value -> value.resource() != null
                            && value.resource().type() == target.type() && value.resource().id().equals(target.id()))
                    .findFirst();
            var reference = batches.advance(supplied.isPresent() ? supplied.get().resource() : target, previous);
            if (type.equals("episode_storyboard_writer") && target.type() == STORYBOARD) {
                addEpisodeBoard(run, node, reference, entries);
            } else {
                addTree(run, reference, true, entries);
            }
        }
        // 祖先版本也要保护；它们只随已授权子对象的命令推进，不能直接改写。
        for (var entry : List.copyOf(entries.values())) {
            ancestors(entry.type(), entry.id(), run.actor(), entries);
        }
        Set<CanvasValueType> create = new LinkedHashSet<>();
        for (var entry : entries.values()) {
            if (entry.writable()) {
                switch (entry.type()) {
                    case SCRIPT -> create.addAll(Set.of(SCRIPT_EPISODE, SCRIPT_SCENE));
                    case SCRIPT_EPISODE -> create.add(SCRIPT_SCENE);
                    case STORYBOARD -> create.addAll(Set.of(STORYBOARD_EPISODE, STORYBOARD_SCENE, SHOT));
                    case STORYBOARD_EPISODE -> create.addAll(Set.of(STORYBOARD_SCENE, SHOT));
                    case STORYBOARD_SCENE -> create.add(SHOT);
                    case ASSET -> create.add(ASSET_ITEM);
                    default -> {
                    }
                }
            }
        }
        if (Set.of("ai_media", "story_to_script", "script_full_parse", "script_episode_parse", "script_assistant",
                "episode_scene_writer", "canvas_asset_extractor", "script_to_storyboard",
                "storyboard_asset_preprocessor").contains(type)) {
            create.addAll(Set.of(ASSET, ASSET_ITEM));
        }
        var config = node.plan().node().config();
        return new CanvasAgentNodeScope(type, entries, create,
                node.plan().environment().has("imageModelId") ? config.path("imageModelId").asLong() : null,
                node.plan().environment().has("videoModelId") ? config.path("videoModelId").asLong() : null,
                config.path("maxImages").asInt(0), config.path("maxVideoSeconds").asInt(0),
                config.path("maxAgentRuns").asInt(16));
    }

    private void addEpisodeBoard(Run run, NodeRun node, CanvasResourceReference reference, Map<String, Entry> entries) {
        var tree = collections.get(run.canvasId(), STORYBOARD, reference.id(), 1, run.actor());
        if (reference.version() == null || reference.version() != tree.root().header().version()) {
            throw CanvasRunException.conflict("分镜容器在确认后变化，请重新预览运行");
        }
        addEntry(STORYBOARD, reference.id(), tree.root().header().version(), true, entries);
        long episode = node.inputs().all().stream()
                .filter(value -> value.resource() != null && value.resource().type() == SCRIPT_EPISODE)
                .findFirst().orElseThrow().resource().id();
        for (var child : tree.descendants()) {
            if (child.header().resourceType() == STORYBOARD_EPISODE
                    && child.data().path("scriptEpisodeId").asLong() == episode) {
                addTree(run, new CanvasResourceReference(STORYBOARD_EPISODE, child.header().resourceId(),
                        (long) child.header().version()), true, entries);
            }
        }
    }

    private void addTree(Run run, CanvasResourceReference ref, boolean writable, Map<String, Entry> entries) {
        var tree = collections.get(run.canvasId(), ref.type(), ref.id(), 3, run.actor());
        if (ref.version() == null || ref.version() != tree.root().header().version()) {
            throw CanvasRunException.conflict("专业智能体输入或目标版本已变化，请重新确认范围");
        }
        addEntry(tree.root().header().resourceType(), tree.root().header().resourceId(), tree.root().header().version(),
                writable, entries);
        for (var child : tree.descendants()) {
            addEntry(child.header().resourceType(), child.header().resourceId(), child.header().version(), writable,
                    entries);
        }
    }

    private void addEntry(CanvasValueType type, long id, int version, boolean writable, Map<String, Entry> entries) {
        String key = key(type, id);
        var old = entries.get(key);
        entries.put(key, new Entry(type, id, version, writable || old != null && old.writable(), false, false));
        if (entries.size() > 2000) {
            throw CanvasRunException.invalid("专业智能体的业务范围超过 2000 个对象，请按分集运行");
        }
    }

    private void ancestors(CanvasValueType type, long id, CanvasActor actor, Map<String, Entry> entries) {
        var resource = resources.get(type, id, actor);
        var data = resource.data();
        switch (type) {
            case SCRIPT_EPISODE -> parent(SCRIPT, data, "scriptId", actor, entries);
            case SCRIPT_SCENE -> parent(SCRIPT_EPISODE, data, "episodeId", actor, entries);
            case STORYBOARD_EPISODE -> parent(STORYBOARD, data, "storyboardId", actor, entries);
            case STORYBOARD_SCENE -> parent(STORYBOARD_EPISODE, data, "episodeId", actor, entries);
            case SHOT -> parent(STORYBOARD_SCENE, data, "storyboardSceneId", actor, entries);
            case ASSET_ITEM -> parent(ASSET, data, "assetId", actor, entries);
            default -> {
            }
        }
    }

    private void parent(CanvasValueType type, ObjectNode data, String field, CanvasActor actor,
                        Map<String, Entry> entries) {
        long id = data.path(field).asLong();
        var parent = resources.get(type, id, actor);
        if (!entries.containsKey(key(type, id))) {
            addEntry(type, id, parent.header().version(), false, entries);
        }
        ancestors(type, id, actor, entries);
    }
}
