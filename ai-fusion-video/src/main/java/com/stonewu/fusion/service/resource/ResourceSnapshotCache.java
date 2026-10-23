package com.stonewu.fusion.service.resource;

import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.resource.model.ResourceHeader;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Component;

/**
 * 授权和版本头不缓存；正文只在对应的事务读取快照内加载。
 */
@Component
@RequiredArgsConstructor
public class ResourceSnapshotCache {
    private final ResourceReadRepository resources;
    private final CanvasJson json;

    @Cacheable(value = "resourceDocument", key = "#header.resourceType().code() + ':' + #header.resourceId() + ':' + #header.version()")
    public String data(ResourceHeader header) {
        return json.write(resources.data(header));
    }
}
