package com.stonewu.fusion.service.media;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.media.model.MediaException;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Component;

@Component
@RequiredArgsConstructor
public class MediaMetadataCache {
    private final MediaResourceRepository resources;
    private final CanvasJson json;

    @Cacheable(value = "mediaResource", key = "#id + ':' + #version")
    public String get(String id, int version) {
        var resource = resources.get(id, false).resource();
        if (resource.version() != version) {
            throw new MediaException(409, "MEDIA_VERSION_CONFLICT", "媒体状态已变化，请刷新");
        }
        return json.write(resource);
    }
}
