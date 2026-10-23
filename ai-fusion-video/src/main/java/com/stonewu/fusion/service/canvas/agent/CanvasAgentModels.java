package com.stonewu.fusion.service.canvas.agent;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class CanvasAgentModels {
    private final CanvasAgentAccess access;
    private final CanvasModelCatalog models;
    private final CanvasJson json;

    public Object query(ObjectNode input, ToolExecutionContext tool) {
        access.require(tool);
        int page = input.path("page").asInt(0);
        if (page < 0 || page > 9999) {
            throw new BusinessException(400, "模型页码无效");
        }
        if (input.hasNonNull("modelId")) {
            var model = models.require(input.path("modelId").asLong());
            return Map.of("status", "success", "model", summary(model), "details",
                    CanvasAgentLibrary.page(json.write(model),
                            input.path("offset").asInt(0), input.path("limit").asInt(8000)));
        }
        String kind = input.path("kind").asText(null);
        var values = models.list().stream().filter(model -> kind == null || model.kind().equals(kind)).toList();
        var audio = models.audioServices();
        return Map.of("status", "success", "page", page, "models",
                values.stream().skip((long) page * 40).limit(40).map(this::summary).toList(),
                "hasMore", (long) (page + 1) * 40 < values.size(), "total", values.size(),
                "audioServices", audio.stream().skip((long) page * 40).limit(40).toList(), "hasMoreAudio",
                (long) (page + 1) * 40 < audio.size());
    }

    private Object summary(CanvasModelCatalog.Option model) {
        var result = new LinkedHashMap<String, Object>();
        result.put("id", model.id());
        result.put("name", model.name());
        result.put("kind", model.kind());
        result.put("resultType", model.resultType());
        result.put("unavailable", model.unavailable());
        result.put("inputs", model.inputs());
        result.put("versionCount", model.versions().size());
        return result;
    }
}
