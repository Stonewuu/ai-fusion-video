package com.stonewu.fusion.service.media;

import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.media.model.MediaSourceType;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;
import java.util.Objects;

@Component
@RequiredArgsConstructor
public class GenerationMediaLinks {
    private final GenerationExecutionContexts executions;
    private final MediaResourceRepository resources;
    private final MediaAccessPolicy access;

    @Transactional(propagation = Propagation.MANDATORY)
    public void beforeWrite(GenerationKind kind, Long taskId, Map<String, String> fields) {
        if (fields.values().stream().noneMatch(MediaReference::managed)) {
            return;
        }
        if (taskId == null) {
            throw MediaException.invalid("生成媒体缺少任务归属");
        }
        executions.running(kind, taskId);
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void capture(GenerationKind kind, long taskId, long itemId, int version, Map<String, String> fields) {
        var references = fields.entrySet().stream().filter(entry -> MediaReference.managed(entry.getValue()))
                .sorted(Map.Entry.comparingByValue()).toList();
        if (references.isEmpty()) {
            return;
        }
        var execution = executions.get(kind, taskId);
        MediaSourceType type = kind == GenerationKind.IMAGE ? MediaSourceType.IMAGE_ITEM : MediaSourceType.VIDEO_ITEM;
        for (var field : references) {
            var resource = resources.get(MediaReference.id(field.getValue()), true).resource();
            access.ready(resource, execution.context().actor());
            if (!Objects.equals(resource.projectId(), execution.projectId())) {
                throw MediaException.invalid("生成结果媒体与任务项目不一致");
            }
            MediaKind expected = kind == GenerationKind.VIDEO && field.getKey()
                    .equals("videoUrl") ? MediaKind.VIDEO : MediaKind.IMAGE;
            if (resource.content().kind() != expected) {
                throw MediaException.invalid("生成条目媒体类型不符");
            }
            resources.bindSource(resource.id(), new MediaSourceReference(type, itemId, version, field.getKey()));
        }
    }
}
