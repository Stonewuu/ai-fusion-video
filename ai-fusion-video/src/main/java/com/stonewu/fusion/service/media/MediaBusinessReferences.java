package com.stonewu.fusion.service.media;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.media.MediaBindingEntity;

import com.stonewu.fusion.mapper.media.MediaBindingMapper;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;

/**
 * 引用授予相应业务对象读者访问权；解绑、删除和撤权立即失效，不改变媒体所有权。
 */
@Component
@RequiredArgsConstructor
public class MediaBusinessReferences {
    private final MediaBindingMapper references;
    private final ResourceReadRepository resources;
    private final ResourceAccessPolicy access;

    public boolean canRead(String mediaId, CanvasActor actor) {
        var bindings = references.selectList(
                        Wrappers.<MediaBindingEntity>lambdaQuery().eq(MediaBindingEntity::getMediaId, mediaId)).stream()
                .map(row -> new Binding(
                        CanvasValueType.fromCode(row.getResourceType()), row.getResourceId(), row.getFieldName()))
                .toList();
        for (Binding binding : bindings) {
            try {
                var header = resources.require(binding.type(), binding.id(), false);
                access.require(header, actor);
                ObjectNode current = resources.data(header);
                var value = current.at("/" + binding.field().replace('.', '/'));
                if (value.isTextual() && MediaReference.of(mediaId).equals(value.textValue())) {
                    return true;
                }
            } catch (CanvasException failure) {
                if (failure.getCode() != 403 && failure.getCode() != 404) {
                    throw failure;
                }
            }
        }
        return false;
    }

    private record Binding(CanvasValueType type, long id, String field) {
    }
}
