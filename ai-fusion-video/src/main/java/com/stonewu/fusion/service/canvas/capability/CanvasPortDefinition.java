package com.stonewu.fusion.service.canvas.capability;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.List;

/**
 * 输出保持具体类型；输入可以明确声明接受的多种内容类型。
 */
public record CanvasPortDefinition(String id, String label, CanvasValueType valueType,
                                   String valueTypeField, boolean collection,
                                   boolean required, int maxConnections, List<CanvasValueType> acceptedTypes) {
    public CanvasPortDefinition(String id, String label, CanvasValueType valueType, String valueTypeField,
                                boolean collection, boolean required, int maxConnections) {
        this(id, label, valueType, valueTypeField, collection, required, maxConnections, List.of());
    }

    public CanvasPortDefinition {
        acceptedTypes = List.copyOf(acceptedTypes);
        int declarations = (valueType == null ? 0 : 1) + (valueTypeField == null ? 0 : 1) + (acceptedTypes.isEmpty() ? 0 : 1);
        if (declarations != 1 || acceptedTypes.stream().distinct().count() != acceptedTypes.size()) {
            throw new IllegalArgumentException("端口必须且只能声明一种类型约束，接受的类型不能重复");
        }
        if (maxConnections < 1 || (!collection && maxConnections != 1)) {
            throw new IllegalArgumentException("端口连接数约束无效");
        }
    }

    public CanvasValueType resolveType(ObjectNode config) {
        var types = resolveTypes(config);
        if (types.size() != 1) {
            throw new IllegalStateException("此输入接受多种素材，请使用 accepts 判断具体类型");
        }
        return types.getFirst();
    }

    public List<CanvasValueType> resolveTypes(ObjectNode config) {
        if (valueType != null) {
            return List.of(valueType);
        }
        if (valueTypeField != null) {
            return List.of(CanvasValueType.fromCode(config.path(valueTypeField).asText()));
        }
        return acceptedTypes;
    }

    public boolean accepts(CanvasValueType type, ObjectNode config) {
        return resolveTypes(config).contains(type);
    }
}
