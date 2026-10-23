package com.stonewu.fusion.service.media;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.entity.media.MediaBindingEntity;

import com.stonewu.fusion.mapper.media.MediaBindingMapper;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.repository.resource.ResourceReadRepository;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.ResourceMutationObserver;
import com.stonewu.fusion.service.resource.model.ResourceChange;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceMutation;
import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

/**
 * 普通页面、Agent、批量创建和正式媒体绑定都经过此处，不能靠提交猜中的媒体 ID 扩权。
 */
@Component
@RequiredArgsConstructor
public class MediaBindingObserver implements ResourceMutationObserver {
    private static final Set<CanvasValueType> TYPES = Set.of(CanvasValueType.PROJECT, CanvasValueType.ASSET,
            CanvasValueType.ASSET_ITEM, CanvasValueType.SHOT, CanvasValueType.STORYBOARD_EPISODE);
    private final MediaBindingMapper bindings;
    private final ResourceReadRepository business;
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;

    @Override
    @Transactional(propagation = Propagation.MANDATORY)
    public void validateAndApply(ResourceMutation mutation, ResourceCommandContext context) {
        List<Binding> pending = new ArrayList<>();
        List<ResourceChange> targets = mutation.changes().stream()
                .filter(change -> TYPES.contains(change.resourceType())).toList();
        for (ResourceChange change : targets) {
            if (change.deleted()) {
                continue;
            }
            var data = business.data(business.require(change.resourceType(), change.resourceId(), false));
            for (String field : MediaSourceFields.fields(change.resourceType()).keySet()) {
                var value = data.at("/" + field.replace('.', '/'));
                if (value.isTextual() && MediaReference.managed(value.textValue())) {
                    var source = MediaSourceFields.get(change.resourceType(), data, field);
                    pending.add(new Binding(change, field, MediaReference.id(source.uri()), source));
                }
            }
        }
        // 验证所有媒体后才改变引用；避免一次请求中先伪造引用、再利用刚得到的权限读取其他媒体。
        Map<String, MediaResource> locked = new HashMap<>();
        for (String id : pending.stream().map(Binding::mediaId).distinct().sorted().toList()) {
            var resource = resources.get(id, true).resource();
            access.ready(resource, context.actor());
            locked.put(id, resource);
        }
        for (Binding binding : pending) {
            if (locked.get(binding.mediaId()).content().kind() != binding.source().kind()) {
                throw MediaException.invalid("媒体类型与业务字段不一致：" + binding.field());
            }
        }
        for (ResourceChange change : targets) {
            bindings.delete(Wrappers.<MediaBindingEntity>lambdaQuery()
                    .eq(MediaBindingEntity::getResourceType, change.resourceType().code())
                    .eq(MediaBindingEntity::getResourceId, change.resourceId()));
        }
        for (Binding binding : pending) {
            bindings.insert(MediaBindingEntity.builder().resourceType(binding.change().resourceType().code())
                    .resourceId(binding.change().resourceId()).fieldName(binding.field()).mediaId(binding.mediaId())
                    .resourceVersion(binding.change().version()).boundBy(context.actor().userId()).build());
        }
    }

    private record Binding(ResourceChange change, String field, String mediaId, MediaSourceFields.Value source) {
    }
}
