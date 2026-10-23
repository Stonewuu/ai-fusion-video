package com.stonewu.fusion.service.canvas.capability;

import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * 节点协议的唯一注册入口。供应商的实际能力仍由生成模型能力服务校验。
 */
@Component
public class CanvasCapabilityRegistry {
    public static final int SCHEMA_VERSION = 1;
    private final Map<CanvasNodeKind, CanvasNodeCapability> definitions;

    public CanvasCapabilityRegistry() {
        Map<CanvasNodeKind, CanvasNodeCapability> registered = new EnumMap<>(CanvasNodeKind.class);
        List<CanvasNodeCapability> entries = new ArrayList<>();
        entries.addAll(CanvasBusinessCapabilities.definitions());
        entries.addAll(CanvasMediaCapabilities.definitions());
        entries.addAll(CanvasUtilityCapabilities.definitions());
        for (CanvasNodeCapability definition : entries) {
            if (registered.putIfAbsent(definition.kind(), definition) != null) {
                throw new IllegalStateException("重复的画布节点能力: " + definition.kind());
            }
        }
        if (registered.size() != CanvasNodeKind.values().length) {
            throw new IllegalStateException("画布节点能力未完整注册");
        }
        definitions = Map.copyOf(registered);
    }

    public CanvasNodeCapability require(CanvasNodeKind kind) {
        CanvasNodeCapability definition = kind == null ? null : definitions.get(kind);
        if (definition == null) {
            throw new CanvasException(CanvasErrorCode.UNKNOWN_NODE_KIND, "节点类型未登记");
        }
        return definition;
    }

    public List<CanvasNodeCapability> list() {
        return List.of(CanvasNodeKind.values()).stream().map(this::require).toList();
    }
}
