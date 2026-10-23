package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import java.util.Map;
import java.util.Set;

/**
 * 专业节点的业务边界；版本仅由同一事务中的已提交业务变更推进。
 */
public record CanvasAgentNodeScope(String agentType, Map<String, Entry> resources, Set<CanvasValueType> createTypes,
                                   Long imageModelId, Long videoModelId, int maxImages, int maxVideoSeconds,
                                   int maxAgentRuns) {
    public record Entry(CanvasValueType type, long id, int version, boolean writable, boolean changed,
                        boolean deleted) {
    }

    public CanvasAgentNodeScope {
        resources = Map.copyOf(resources);
        createTypes = Set.copyOf(createTypes);
    }

    public static String key(CanvasValueType type, long id) {
        return type.code() + ":" + id;
    }

    public CanvasAgentNodeScope withResources(Map<String, Entry> values) {
        return new CanvasAgentNodeScope(agentType, values, createTypes, imageModelId, videoModelId, maxImages,
                maxVideoSeconds, maxAgentRuns);
    }
}
