package com.stonewu.fusion.service.canvas.template;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.CanvasMcpConfigMedia;
import com.stonewu.fusion.service.canvas.collaboration.CanvasGraphCopies;
import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkSources;
import com.stonewu.fusion.service.canvas.collaboration.CanvasProjectForkSources.Entry;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.resource.ResourceForkReferences;
import com.stonewu.fusion.service.resource.ResourceForkReferences.Key;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;

import java.nio.charset.StandardCharsets;
import java.util.*;

import static com.stonewu.fusion.service.canvas.template.CanvasTemplateData.*;

@Component
@RequiredArgsConstructor
public class CanvasTemplateExport {
    private final CanvasProjectForkSources sources;
    private final CanvasTemplateProviders providers;
    private final CanvasTemplateResourceValidation fields;
    private final CanvasMcpConfigMedia mcpMedia;
    private final CanvasJson json;
    private final CanvasTemplateValidation validation;

    @Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ)
    public Template create(String canvas, long revision, CanvasActor actor) {
        var source = sources.read(canvas, revision, actor, false);
        if (!source.warnings().isEmpty()) {
            throw invalid("模板中的旧素材尚未登记，请先导入素材库：" + source.warnings().getFirst());
        }
        String namespace = uuid("template:" + json.hash(source));
        var normalization = new Normalization(namespace, source);
        var copied = CanvasGraphCopies.copy(source.manifest().graph(), namespace);
        Map<String, Provider> slots = new LinkedHashMap<>();
        List<CanvasNode> nodes = new ArrayList<>();
        for (var node : copied.nodes()) {
            ObjectNode config = (ObjectNode) normalization.media(
                    mcpMedia.rewrite(node, id -> MediaReference.of(normalization.mediaId(id))), null);
            if (node.kind() == CanvasNodeKind.SELECTION) {
                config.remove("selectedOutputId");
            }
            if (node.kind().storedMedia() && node.config().hasNonNull("resourceId")) {
                config.put("resourceId", normalization.mediaId(node.config().path("resourceId").asText()));
            }
            var draft = new CanvasNode(node.id(), node.kind(), 1, node.title(), node.layout(), config, node.resource(),
                    0, 0);
            for (var found : providers.references(draft)) {
                String key = found.kind() + ":" + found.id().asText();
                var previous = slots.get(key);
                List<Use> uses = new ArrayList<>(previous == null ? List.of() : previous.uses());
                uses.add(found.use());
                slots.put(key, new Provider(uuid(namespace + ":provider:" + key), found.kind(),
                        node.title() + " · " + providerLabel(found.kind()), List.copyOf(uses)));
                var target = providers.config(draft, found.use().batch());
                target.remove(found.use().field());
                if (found.kind() == ProviderKind.MCP) {
                    target.put("contractVersion", 0);
                }
            }
            var ref = node.resource();
            if (ref != null) {
                ref = new CanvasResourceReference(ref.type(), normalization.id(ref.type(), ref.id()),
                        normalization.version(ref.type(), ref.id()));
            }
            nodes.add(new CanvasNode(node.id(), node.kind(), 1, node.title(), node.layout(), config, ref, 0, 0));
        }
        var edges = copied.edges().stream()
                .map(edge -> new CanvasEdge(edge.id(), edge.kind(), edge.sourceNodeId(), edge.sourcePort(),
                        edge.targetNodeId(), edge.targetPort(), edge.inputIndex(), null, 0)).toList();
        Map<String, Value> fixed = new LinkedHashMap<>();
        for (int index = 0; index < copied.edges().size(); index++) {
            String original = source.manifest().graph().edges().get(index).id();
            var value = source.manifest().fixedOutputs().get(original);
            if (value != null) {
                fixed.put(copied.edges().get(index).id(), normalization.value(value));
            }
        }
        Long project = source.manifest().graph().projectId() == null ? null : normalization.id(CanvasValueType.PROJECT,
                source.manifest().graph().projectId());
        var graph = new CanvasGraph(1, project, List.copyOf(nodes), edges);
        var objects = source.objects().stream()
                .map(entry -> new Entry(entry.type(), normalization.id(entry.type(), entry.id()),
                        normalization.business(entry.type(), entry.data()))).toList();
        var files = source.media().stream()
                .map(file -> new Media(normalization.mediaId(file.id()), file.name(), file.content())).toList();
        var result = new Template(SCHEMA, VERSION, source.manifest().title(), graph, objects,
                source.primaryScript() == null ? null : normalization.id(CanvasValueType.SCRIPT,
                        source.primaryScript()),
                source.primaryStoryboard() == null ? null : normalization.id(CanvasValueType.STORYBOARD,
                        source.primaryStoryboard()), files, List.copyOf(slots.values()), Map.copyOf(fixed));
        if (json.write(result).getBytes(StandardCharsets.UTF_8).length > MAX_BYTES) {
            throw invalid("模板超过 16 MiB，请拆分后导出");
        }
        validation.validate(result);
        return result;
    }

    private final class Normalization {
        private final String namespace;
        private final Map<Key, Long> ids = new LinkedHashMap<>();
        private final Map<Key, Long> versions = new LinkedHashMap<>();
        private final Map<String, String> media = new LinkedHashMap<>();

        Normalization(String namespace, CanvasProjectForkSources.Frozen source) {
            this.namespace = namespace;
            for (var entry : source.objects()) {
                var key = new Key(entry.type(), entry.id());
                ids.put(key, (long) ids.size() + 1);
                versions.put(key, entry.data().path("version").longValue());
            }
            source.media().forEach(file -> media.put(file.id(), uuid(namespace + ":media:" + file.id())));
        }

        long id(CanvasValueType type, long id) {
            Long value = ids.get(new Key(type, id));
            if (value == null) {
                throw invalid("模板缺少业务依赖：" + type.code() + "/" + id);
            }
            return value;
        }

        long version(CanvasValueType type, long id) {
            return versions.get(new Key(type, id));
        }

        String mediaId(String id) {
            var value = media.get(id);
            if (value == null) {
                throw invalid("模板缺少已授权素材");
            }
            return value;
        }

        ObjectNode business(CanvasValueType type, ObjectNode original) {
            var data = ResourceForkReferences.rewrite(type, original, this::id);
            long local = id(type, original.path("id").longValue());
            data.put("id", local);
            original.path("_exportMediaBindings").fields().forEachRemaining(entry -> {
                String[] path = entry.getKey().split("\\.");
                ObjectNode parent = data;
                for (int index = 0; index < path.length - 1; index++) {
                    parent = (ObjectNode) parent.path(path[index]);
                }
                parent.set(path[path.length - 1], entry.getValue());
            });
            if (type == CanvasValueType.STORYBOARD_EPISODE) {
                data.put("deletedId", data.path("deleted").asBoolean() ? local : 0);
            }
            ObjectNode result = fields.clean(type, (ObjectNode) media(data, null));
            if (original.path("_canvasVideos").isArray()) {
                var videos = result.putArray("_canvasVideos");
                original.path("_canvasVideos")
                        .forEach(item -> videos.add(json.object(value(json.read(item.toString(), Value.class)))));
            }
            return result;
        }

        Value value(Value original) {
            var ref = original.resource();
            ObjectNode data = null;
            if (ref != null) {
                data = business(ref.type(), original.data());
                ref = new CanvasResourceReference(ref.type(), id(ref.type(), ref.id()), ref.version());
            }
            String mediaId = original.mediaId() == null ? null : mediaId(original.mediaId());
            return new Value(uuid(namespace + ":value:" + original.id()), original.type(), original.text(), mediaId,
                    mediaId == null ? null : 0, ref, data, null);
        }

        JsonNode media(JsonNode original, String field) {
            if (original == null || original.isNull()) {
                return original;
            }
            if (original.isTextual()) {
                String value = original.asText();
                if (MediaReference.managed(value)) {
                    return TextNode.valueOf(MediaReference.of(mediaId(MediaReference.id(value))));
                }
                return original;
            }
            if (original.isObject()) {
                var result = ((ObjectNode) original).deepCopy();
                original.fields()
                        .forEachRemaining(entry -> result.set(entry.getKey(), media(entry.getValue(), entry.getKey())));
                return result;
            }
            if (original.isArray()) {
                var result = ((ArrayNode) original).arrayNode();
                original.forEach(item -> result.add(media(item, field)));
                return result;
            }
            return original;
        }
    }

    private static String uuid(String text) {
        return UUID.nameUUIDFromBytes(text.getBytes(StandardCharsets.UTF_8)).toString();
    }

    private static String providerLabel(ProviderKind kind) {
        return switch (kind) {
            case TEXT_MODEL -> "文本模型";
            case IMAGE_MODEL -> "图片模型";
            case VIDEO_MODEL -> "视频模型";
            case COMFYUI_MODEL -> "工作流模型";
            case AUDIO_API -> "配乐服务";
            case WORKFLOW -> "工作流版本";
            case MCP -> "MCP 操作";
        };
    }

    private static BusinessException invalid(String text) {
        return new BusinessException(400, text);
    }
}
