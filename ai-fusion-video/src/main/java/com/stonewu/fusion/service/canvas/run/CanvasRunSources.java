package com.stonewu.fusion.service.canvas.run;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasResourceCollectionService;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.media.*;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceSnapshot;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Component
@RequiredArgsConstructor
public class CanvasRunSources {
    private final MediaResourceQueries media;
    private final MediaSourceQueries mediaSources;
    private final MediaAccessPolicy mediaAccess;
    private final MediaContentReader contents;
    private final ResourceQueryService resources;
    private final CanvasResourceCollectionService collections;
    private final CanvasJson json;

    public Ports freeze(CanvasDocument document, CanvasNode node, CanvasActor actor, Set<String> usedPorts) {
        if (node.kind() == CanvasNodeKind.TEXT || node.kind() == CanvasNodeKind.STORY) {
            return new Ports(Map.of("text", List.of(text(node.config().path("content").asText("")))));
        }
        if (Set.of(CanvasNodeKind.IMAGE, CanvasNodeKind.VIDEO, CanvasNodeKind.AUDIO, CanvasNodeKind.TEXT_FILE,
                CanvasNodeKind.DOCUMENT).contains(node.kind())) {
            String id = node.config().path("resourceId").asText();
            var value = mediaValue(id, actor, document.projectId());
            if (node.kind() == CanvasNodeKind.TEXT_FILE) {
                value = value(CanvasValueType.TEXT, readText(id, actor), value.mediaId(), value.mediaVersion(), null,
                        value.data());
            }
            return new Ports(Map.of("media", List.of(value)));
        }
        if (node.resource() == null) {
            return Ports.empty();
        }
        var snapshot = resources.get(node.resource().type(), node.resource().id(), actor);
        if (snapshot.header().projectId() != null && !Objects.equals(snapshot.header().projectId(),
                document.projectId())) {
            throw CanvasRunException.invalid("业务对象不属于当前画布项目");
        }
        var data = snapshot.data();
        Map<String, List<Value>> output = new LinkedHashMap<>();
        String port = switch (node.kind()) {
            case PROJECT -> "project";
            case SCRIPT -> "script";
            case SCRIPT_EPISODE, STORYBOARD_EPISODE -> "episode";
            case SCRIPT_SCENE, STORYBOARD_SCENE -> "scene";
            case STORYBOARD -> "storyboard";
            case SHOT -> "shot";
            case ASSET -> "asset";
            case ASSET_ITEM -> "assetItem";
            default -> throw CanvasRunException.invalid("未登记的业务来源节点");
        };
        output.put(port, List.of(business(snapshot)));
        switch (node.kind()) {
            case PROJECT -> {
                output.put("style", List.of(text(data.path("artStyleDescription").asText(""))));
                mediaField(output, "image", snapshot, "artStyleImageUrl", usedPorts, actor, document.projectId());
            }
            case SCRIPT -> output.put("text", List.of(text(data.path("rawContent").asText(""))));
            case SCRIPT_EPISODE -> output.put("text", List.of(text(data.path("rawContent").asText(""))));
            case SCRIPT_SCENE -> output.put("text", List.of(text(sceneText(data))));
            case SHOT -> {
                output.put("prompt", List.of(text(data.path("videoPrompt").asText(""))));
                mediaField(output, "firstFrame", snapshot, "firstFrameImageUrl", usedPorts, actor,
                        document.projectId());
                mediaField(output, "lastFrame", snapshot, "lastFrameImageUrl", usedPorts, actor, document.projectId());
                mediaField(output, "video", snapshot, videoField(data), usedPorts, actor, document.projectId());
            }
            case ASSET_ITEM -> {
                mediaField(output, "image", snapshot, "imageUrl", usedPorts, actor, document.projectId());
                if (data.path("properties").isObject()) {
                    mediaField(output, "video", snapshot, "properties.videoUrl", usedPorts, actor,
                            document.projectId());
                }
            }
            case STORYBOARD_EPISODE, STORYBOARD_SCENE, ASSET -> {
                int depth = node.kind() == CanvasNodeKind.STORYBOARD_EPISODE ? 2 : 1;
                var childType = node.kind() == CanvasNodeKind.ASSET ? CanvasValueType.ASSET_ITEM : CanvasValueType.SHOT;
                var values = collections.get(document.id(), snapshot.header().resourceType(),
                                snapshot.header().resourceId(), depth, actor)
                        .descendants().stream().filter(entry -> entry.header().resourceType() == childType)
                        .map(entry -> business(new ResourceSnapshot(entry.header(), entry.data()))).toList();
                output.put(node.kind() == CanvasNodeKind.ASSET ? "items" : "shots", values);
                if (node.kind() == CanvasNodeKind.STORYBOARD_EPISODE && usedPorts.contains("_canvasVideos")) {
                    var videos = values.stream().map(shot -> {
                        String field = videoField(shot.data());
                        String reference = shot.data().path(field).asText("");
                        if (reference.isBlank()) {
                            throw CanvasRunException.invalid(
                                    "合成需要每个镜头都有视频，请检查镜头 " + shot.resource().id());
                        }
                        return sourceMedia(shot.resource(), field, reference, actor, document.projectId());
                    }).toList();
                    var expanded = data.deepCopy();
                    expanded.set("_canvasVideos", json.object(Map.of("values", videos)).path("values"));
                    output.put("episode", List.of(value(snapshot.header().resourceType(), null, null, null,
                            new CanvasResourceReference(snapshot.header().resourceType(),
                                    snapshot.header().resourceId(), (long) snapshot.header().version()), expanded)));
                }
            }
            default -> {
            }
        }
        return new Ports(output);
    }

    public Value text(String text) {
        return value(CanvasValueType.TEXT, text, null, null, null, null);
    }

    public boolean unchanged(NodePlan plan, CanvasDocument document, CanvasNode node, CanvasActor actor,
                             Set<String> usedPorts) {
        if (node.kind() != CanvasNodeKind.TEXT_FILE) {
            return json.hash(freeze(document, node, actor, usedPorts)).equals(json.hash(plan.sourceOutputs()));
        }
        var previous = plan.sourceOutputs().at("media").getFirst();
        var current = mediaValue(node.config().path("resourceId").asText(), actor, document.projectId());
        return Objects.equals(previous.mediaId(), current.mediaId()) && Objects.equals(previous.mediaVersion(),
                current.mediaVersion());
    }

    public Value mediaValue(String id, CanvasActor actor, Long projectId) {
        var resource = media.get(id, actor);
        mediaAccess.ready(resource, actor);
        if (resource.projectId() != null && !Objects.equals(resource.projectId(), projectId)) {
            throw CanvasRunException.invalid("媒体来自其他项目，请先导入当前项目");
        }
        return value(CanvasValueType.fromCode(resource.content().kind().name().toLowerCase()), null,
                resource.id(), resource.version(), null, json.object(resource));
    }

    public void authorize(Value value, CanvasActor actor, boolean exactMediaVersion) {
        if (value.resource() != null) {
            resources.authorizeHistorical(value.resource().type(), value.resource().id(), actor);
        }
        if (value.mediaId() != null) {
            var current = media.get(value.mediaId(), actor);
            mediaAccess.ready(current, actor);
            if (exactMediaVersion && !Objects.equals(value.mediaVersion(), current.version())) {
                throw CanvasRunException.conflict("已固定的媒体版本发生变化，请重新预览运行计划");
            }
        }
        if (value.data() != null && value.data().path("_canvasVideos").isArray()) {
            value.data().path("_canvasVideos")
                    .forEach(nested -> authorize(json.read(nested.toString(), Value.class), actor, exactMediaVersion));
        }
    }

    public boolean visible(Value value, CanvasActor actor) {
        if (value.resource() != null && !resources.canReadHistorical(value.resource().type(), value.resource().id(),
                actor)) {
            return false;
        }
        if (value.mediaId() != null && media.visible(value.mediaId(), actor)
                .filter(item -> "READY".equals(item.state())).isEmpty()) {
            return false;
        }
        if (value.data() != null && value.data().path("_canvasVideos").isArray()) {
            for (var nested : value.data().path("_canvasVideos")) {
                if (!visible(json.read(nested.toString(), Value.class), actor)) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean visible(NodeRun node, CanvasActor actor) {
        return node.outputs().all().stream().allMatch(value -> visible(value, actor))
                && (node.inputs() == null || node.inputs().all().stream().allMatch(value -> visible(value, actor)))
                && node.plan().sourceOutputs().all().stream().allMatch(value -> visible(value, actor))
                && node.plan().pinned().values().stream().allMatch(value -> visible(value, actor));
    }

    public Value produced(Value value, String nodeRunId, String port, int index) {
        String id = UUID.nameUUIDFromBytes((nodeRunId + ":" + port + ":" + index).getBytes(StandardCharsets.UTF_8))
                .toString();
        return new Value(id, value.type(), value.text(), value.mediaId(), value.mediaVersion(), value.resource(),
                value.data(), nodeRunId);
    }

    public Value business(ResourceSnapshot snapshot) {
        var header = snapshot.header();
        return value(header.resourceType(), null, null, null,
                new CanvasResourceReference(header.resourceType(), header.resourceId(), (long) header.version()),
                snapshot.data());
    }

    private Value value(CanvasValueType type, String text, String mediaId, Integer version,
                        CanvasResourceReference reference, ObjectNode data) {
        var raw = new Value("", type, text, mediaId, version, reference, data, null);
        String id = UUID.nameUUIDFromBytes(json.hash(raw).getBytes(StandardCharsets.UTF_8)).toString();
        return new Value(id, type, text, mediaId, version, reference, data, null);
    }

    private void mediaField(Map<String, List<Value>> result, String port, ResourceSnapshot snapshot, String field,
                            Set<String> used, CanvasActor actor, Long projectId) {
        var current = snapshot.data().at("/" + field.replace('.', '/'));
        String reference = current.asText("");
        if (reference.isBlank()) {
            return;
        }
        if (!MediaReference.managed(reference) && !used.contains(port)) {
            return;
        }
        var header = snapshot.header();
        var source = new CanvasResourceReference(header.resourceType(), header.resourceId(), (long) header.version());
        result.put(port, List.of(sourceMedia(source, field, reference, actor, projectId)));
    }

    private Value sourceMedia(CanvasResourceReference source, String field, String reference, CanvasActor actor,
                              Long projectId) {
        if (MediaReference.managed(reference)) {
            return mediaValue(MediaReference.id(reference), actor, projectId);
        }
        var observed = mediaSources.resolve(
                new MediaSourceReference(source.type(), source.id(), source.version().intValue(), field),
                ResourceCommandContext.user(actor, null));
        var imported = media.imported(observed, actor).orElseThrow(() -> CanvasRunException.invalid(
                "此节点的历史素材尚未导入，请在对象素材或素材面板中完成导入后重新预览运行计划"));
        return mediaValue(imported.id(), actor, projectId);
    }

    private String videoField(ObjectNode shot) {
        return shot.path("videoUrl").asText("").isBlank() ? "generatedVideoUrl" : "videoUrl";
    }

    private String readText(String id, CanvasActor actor) {
        try {
            var path = Files.createTempFile("afv-canvas-text-", ".txt");
            try {
                contents.copy(id, actor, path, 512 * 1024);
                String result = Files.readString(path, StandardCharsets.UTF_8);
                if (result.length() > 128000) {
                    throw CanvasRunException.invalid("文本文件超过 128000 字符，请先拆分");
                }
                return result;
            } finally {
                Files.deleteIfExists(path);
            }
        } catch (IOException failure) {
            throw CanvasRunException.invalid("文本文件无法读取为 UTF-8 文本");
        }
    }

    private String sceneText(ObjectNode data) {
        StringBuilder text = new StringBuilder();
        for (String field : List.of("sceneHeading", "sceneDescription", "dialogues")) {
            var value = data.path(field);
            if (!value.isMissingNode() && !value.isNull()) {
                if (!text.isEmpty()) {
                    text.append('\n');
                }
                text.append(value.isTextual() ? value.asText() : value.toString());
            }
        }
        return text.toString();
    }
}
