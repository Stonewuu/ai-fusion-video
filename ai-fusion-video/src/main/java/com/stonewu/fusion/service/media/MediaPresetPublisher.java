package com.stonewu.fusion.service.media;

import com.stonewu.fusion.mapper.media.MediaPresetBindingMapper;

import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.system.SystemConfigService;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

@Component
@RequiredArgsConstructor
public class MediaPresetPublisher {
    private final MediaPresetBindingMapper bindings;
    private final MediaResourceQueries resources;
    private final MediaAccessPolicy access;
    private final SystemConfigService config;

    @Transactional
    public String publish(String key, String id, ResourceCommandContext context) {
        var resource = resources.get(id, context.actor());
        access.edit(resource, context);
        if (resource.projectId() != null || resource.content().kind() != MediaKind.IMAGE) {
            throw MediaException.invalid("内置预设只接受独立图片");
        }
        String reference = MediaReference.of(id);
        config.setValue("art_preset_url:" + key, reference);
        bindings.publish(key, id, context.actor().userId());
        return reference;
    }
}
