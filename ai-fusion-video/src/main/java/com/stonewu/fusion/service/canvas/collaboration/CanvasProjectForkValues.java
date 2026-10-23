package com.stonewu.fusion.service.canvas.collaboration;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.ResourceForkReferences;
import com.stonewu.fusion.service.resource.ResourceForkReferences.Key;

import java.util.Map;

/**
 * 每次复制独享的编号映射；不会在 Spring 单例中保留用户状态。
 */
public final class CanvasProjectForkValues {
    private final long project;
    private final CanvasActor actor;
    private final Map<Key, Long> objects;
    private final Map<String, MediaResource> media;
    private final CanvasJson json;

    public CanvasProjectForkValues(long project, CanvasActor actor, Map<Key, Long> objects,
                                   Map<String, MediaResource> media, CanvasJson json) {
        this.project = project;
        this.actor = actor;
        this.objects = objects;
        this.media = media;
        this.json = json;
    }

    public long id(CanvasValueType type, long id) {
        Long result = objects.get(new Key(type, id));
        if (result == null) {
            throw new BusinessException(409, "副本缺少关联对象：" + type.code() + "/" + id);
        }
        return result;
    }

    public ObjectNode business(CanvasValueType type, ObjectNode source) {
        var data = ResourceForkReferences.rewrite(type, source, this::id);
        if (data.has("projectId")) {
            data.put("projectId", project);
        }
        if (data.has("ownerType")) {
            data.put("ownerType", actor.teamId() == null ? 1 : 2);
        }
        if (data.has("ownerId")) {
            data.put("ownerId", actor.teamId() == null ? actor.userId() : actor.teamId());
        }
        if (data.has("userId")) {
            data.put("userId", actor.userId());
        }
        if (data.has("scope")) {
            data.put("scope", 2);
        }
        var bindings = data.path("_exportMediaBindings");
        bindings.fields().forEachRemaining(entry -> {
            var field = entry.getKey().split("\\.");
            ObjectNode parent = data;
            for (int index = 0; index < field.length - 1; index++) {
                parent = (ObjectNode) parent.path(field[index]);
            }
            parent.set(field[field.length - 1], entry.getValue());
        });
        data.remove("_exportMediaBindings");
        return (ObjectNode) media(data, null);
    }

    public Value value(Value source) {
        var resource = source.resource();
        ObjectNode data = source.data() == null ? null : source.data().deepCopy();
        if (resource != null) {
            data = business(resource.type(), data);
            long id = id(resource.type(), resource.id());
            resource = new CanvasResourceReference(resource.type(), id, resource.version());
            data.put("id", id);
        }
        if (data != null && data.path("_canvasVideos").isArray()) {
            ArrayNode videos = data.arrayNode();
            for (var item : source.data().path("_canvasVideos")) {
                videos.add(json.object(value(json.read(item.toString(), Value.class))));
            }
            data.set("_canvasVideos", videos);
        }
        String mediaId = source.mediaId();
        Integer version = source.mediaVersion();
        if (mediaId != null) {
            var copy = requireMedia(mediaId);
            mediaId = copy.id();
            version = copy.version();
            data = json.object(copy);
        } else if (resource == null && data != null) {
            data = (ObjectNode) media(data, null);
        }
        return new Value(source.id(), source.type(), source.text(), mediaId, version, resource, data,
                source.sourceNodeRunId());
    }

    public JsonNode media(JsonNode source, String field) {
        if (source == null || source.isNull()) {
            return source;
        }
        if (source.isTextual()) {
            String value = source.asText();
            if (MediaReference.managed(value)) {
                return TextNode.valueOf(MediaReference.of(requireMedia(MediaReference.id(value)).id()));
            }
            return source;
        }
        if (source.isObject()) {
            ObjectNode result = ((ObjectNode) source).deepCopy();
            source.fields()
                    .forEachRemaining(entry -> result.set(entry.getKey(), media(entry.getValue(), entry.getKey())));
            return result;
        }
        if (source.isArray()) {
            ArrayNode result = ((ArrayNode) source).arrayNode();
            source.forEach(item -> result.add(media(item, field)));
            return result;
        }
        return source;
    }

    private MediaResource requireMedia(String id) {
        var result = media.get(id);
        if (result == null) {
            throw new BusinessException(409, "副本缺少已授权媒体：" + id);
        }
        return result;
    }

    public String mediaReference(String id) {
        return MediaReference.of(requireMedia(id).id());
    }

    public String mediaId(String id) {
        return requireMedia(id).id();
    }
}
