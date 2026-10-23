package com.stonewu.fusion.service.media;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.resource.ResourceRestorePlanner;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * 恢复预览与实际提交都检查旧媒体；历史记录本身不授予文件读取权限。
 */
@Component
@RequiredArgsConstructor
public class MediaRestoreValidation {
    private static final Set<CanvasValueType> TYPES = Set.of(CanvasValueType.PROJECT, CanvasValueType.ASSET,
            CanvasValueType.ASSET_ITEM, CanvasValueType.SHOT, CanvasValueType.STORYBOARD_EPISODE);
    private final MediaResourceRepository media;
    private final MediaAccessPolicy access;

    public void validate(List<ResourceRestorePlanner.Change> changes, CanvasActor actor, boolean lock) {
        Map<String, MediaKind> required = new TreeMap<>();
        for (var change : changes) {
            if (!TYPES.contains(change.resourceType()) || change.proposed().path("deleted").asBoolean()) {
                continue;
            }
            for (String field : MediaSourceFields.fields(change.resourceType()).keySet()) {
                var value = change.proposed().at("/" + field.replace('.', '/'));
                if (!value.isTextual() || !MediaReference.managed(value.asText())) {
                    continue;
                }
                var source = MediaSourceFields.get(change.resourceType(), change.proposed(), field);
                String id = MediaReference.id(source.uri());
                var previous = required.putIfAbsent(id, source.kind());
                if (previous != null && previous != source.kind()) {
                    throw MediaException.invalid("恢复内容中同一素材的类型不一致");
                }
            }
        }
        for (var entry : required.entrySet()) {
            var resource = media.get(entry.getKey(), lock).resource();
            access.ready(resource, actor);
            if (resource.content().kind() != entry.getValue()) {
                throw MediaException.invalid("恢复所需素材与业务字段类型不一致");
            }
        }
    }
}
