package com.stonewu.fusion.service.resource;

import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.stereotype.Component;

import java.util.Map;

@Component
@RequiredArgsConstructor
public class AssetCommandSupport {
    private final ResourceReadRepository resources;
    private final ResourceAccessPolicy access;
    private final CanvasAccessPolicy projects;

    public ResourceHeader authorize(CanvasValueType type, long id, ResourceCommandContext context,
                                    boolean deletedReplay) {
        var header = resources.find(type, id, false, deletedReplay).orElseThrow(() -> new CanvasException(404,
                CanvasErrorCode.RESOURCE_DELETED, "资产不存在或已删除", Map.of()));
        access.requireEdit(header, context.actor());
        return header;
    }

    public void authorizeCreate(Long projectId, ResourceCommandContext context) {
        projects.requireActorScope(context.actor());
        if (projectId != null) {
            projects.requireProjectEditor(projectId, context.actor());
        }
    }

    public static long id(JsonNode value, String field) {
        if (value == null || !value.isIntegralNumber() || !value.canConvertToLong() || value.longValue() <= 0) {
            throw AssetFields.invalid(field + " 必须为正整数");
        }
        return value.longValue();
    }

    public static int version(JsonNode value, String field) {
        if (value == null || !value.isIntegralNumber() || !value.canConvertToInt() || value.intValue() < 0) {
            throw AssetFields.invalid(field + " 必须为非负整数版本");
        }
        return value.intValue();
    }

    @CacheEvict(value = {"asset", "assetItem", "resourceDocument"}, allEntries = true)
    public void evict() {
    }
}
