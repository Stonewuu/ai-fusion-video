package com.stonewu.fusion.service.canvas.delivery;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.MediaSourceQueries;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * 只解析已登记字段与已显式导入的版本映射，不因导出发起下载或改写原业务 URL。
 */
@Component
@RequiredArgsConstructor
public class CanvasExportMedia {
    private final MediaSourceQueries sources;
    private final MediaResourceQueries media;
    private final CanvasJson json;

    public Value resolve(Value value, CanvasActor actor, Set<String> ids, Set<String> warnings) {
        if (value.mediaId() != null) {
            ids.add(value.mediaId());
        }
        scan(json.object(value), ids, null);
        if (value.data() != null && value.data().path("_canvasVideos").isArray()) {
            value.data().path("_canvasVideos")
                    .forEach(item -> resolve(json.read(item.toString(), Value.class), actor, ids, warnings));
        }
        if (value.resource() == null) {
            return value;
        }
        var reference = value.resource();
        Map<String, String> bindings = new LinkedHashMap<>();
        for (String field : sources.fields(reference.type()).stream().sorted().toList()) {
            JsonNode stored = value.data();
            for (String part : field.split("\\.")) {
                stored = stored.path(part);
            }
            if (!stored.isTextual() || stored.asText().isBlank() || MediaReference.managed(stored.asText())) {
                continue;
            }
            var source = sources.frozen(
                    new MediaSourceReference(reference.type(), reference.id(), Math.toIntExact(reference.version()),
                            field),
                    value.data(), actor);
            var imported = media.imported(source, actor);
            if (imported.isPresent()) {
                ids.add(imported.get().id());
                bindings.put(field, MediaReference.of(imported.get().id()));
            } else {
                warnings.add(reference.type()
                        .code() + "/" + reference.id() + " 的 " + field + " 尚未导入素材库，未包含在媒体包中");
            }
        }
        if (bindings.isEmpty()) {
            return value;
        }
        var data = value.data().deepCopy();
        data.set("_exportMediaBindings", json.object(bindings));
        return new Value(value.id(), value.type(), value.text(), value.mediaId(), value.mediaVersion(),
                value.resource(), data, value.sourceNodeRunId());
    }

    public void scan(JsonNode node, Set<String> ids, String field) {
        if (node == null || node.isNull()) {
            return;
        }
        if (node.isTextual()) {
            String value = node.asText();
            if (MediaReference.managed(value)) {
                ids.add(MediaReference.id(value));
            }
        } else if (node.isObject()) {
            node.fields().forEachRemaining(entry -> scan(entry.getValue(), ids, entry.getKey()));
        } else if (node.isArray()) {
            node.forEach(child -> scan(child, ids, field));
        }
    }
}
