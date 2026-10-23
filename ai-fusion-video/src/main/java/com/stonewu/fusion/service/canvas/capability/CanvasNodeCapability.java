package com.stonewu.fusion.service.canvas.capability;

import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.List;
import java.util.Map;

public record CanvasNodeCapability(CanvasNodeKind kind, String label, String category,
                                   String executor, int schemaVersion, boolean requiresProject,
                                   CanvasValueType resourceType,
                                   Map<String, CanvasConfigField> configFields,
                                   List<CanvasPortDefinition> inputs,
                                   List<CanvasPortDefinition> outputs) {
    public CanvasNodeCapability {
        configFields = Map.copyOf(configFields);
        inputs = List.copyOf(inputs);
        outputs = List.copyOf(outputs);
        if (outputs.stream().anyMatch(port -> !port.acceptedTypes().isEmpty())) {
            throw new IllegalArgumentException("输出端口必须保留具体类型，不能声明多种接受类型");
        }
    }
}
