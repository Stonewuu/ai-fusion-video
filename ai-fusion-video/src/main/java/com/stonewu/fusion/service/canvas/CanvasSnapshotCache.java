package com.stonewu.fusion.service.canvas;

import com.stonewu.fusion.repository.canvas.CanvasGraphRepository;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Component;

/**
 * 只缓存特定已提交版本的内容，授权与文档头始终实时查询。
 */
@Component
@RequiredArgsConstructor
public class CanvasSnapshotCache {
    private final CanvasGraphRepository graphs;
    private final CanvasJson json;

    @Cacheable(value = "canvasGraph", key = "#document.id() + ':' + #document.revision()")
    public String graph(CanvasDocument document) {
        // 缓存 JSON 字符串，避免让 Redis 全局多态反序列化器接受任意 JsonNode 类型。
        return json.write(graphs.load(document));
    }

    @CacheEvict(value = "canvasGraph", key = "#document.id() + ':' + #document.revision()")
    public void evict(CanvasDocument document) {
    }
}
