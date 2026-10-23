package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.model.ResourceChange;
import com.stonewu.fusion.service.resource.model.ResourceHeader;

import java.util.List;
import java.util.Map;
import java.util.UUID;

public final class ResourceMutations {
    private ResourceMutations() {
    }

    public static void requireVersion(ResourceHeader resource, int expected) {
        if (resource.version() != expected) {
            throw new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                    "业务内容已被其他操作修改",
                    Map.of("resourceType", resource.resourceType().code(), "resourceId", resource.resourceId(),
                            "currentVersion", resource.version()));
        }
        if (expected == Integer.MAX_VALUE) {
            throw new CanvasException(CanvasErrorCode.INVALID_COMMAND, "业务版本达到上限");
        }
    }

    public static ResourceChange change(CanvasValueType type, long id, Long projectId, int version, boolean deleted,
                                        List<String> fields) {
        return new ResourceChange(UUID.randomUUID().toString(), type, id, projectId, version, deleted, fields);
    }
}
