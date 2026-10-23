package com.stonewu.fusion.service.generation.job;

import cn.hutool.json.JSONObject;
import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.generation.VideoMediaRules;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaFacts;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
@RequiredArgsConstructor
public class GenerationInputVersions {
    private final MediaResourceQueries resources;
    private final MediaAccessPolicy access;
    private final ReferenceImageTransportService references;

    public void freeze(ImageTask task, CanvasActor actor) {
        var values = new ArrayList<>(references.parseJsonInputs(task.getRefImageUrls(), "refImageUrls"));
        collect(task.getWorkflowInputs(), values);
        task.setInputMediaVersions(freeze(values, task.getInputMediaVersions(), actor));
    }

    public void freeze(VideoTask task, CanvasActor actor) {
        List<String> values = new ArrayList<>();
        values.add(task.getFirstFrameImageUrl());
        values.add(task.getLastFrameImageUrl());
        values.addAll(references.parseJsonInputs(task.getReferenceImageUrls(), "referenceImageUrls"));
        values.addAll(references.parseJsonInputs(task.getReferenceVideoUrls(), "referenceVideoUrls"));
        values.addAll(references.parseJsonInputs(task.getReferenceAudioUrls(), "referenceAudioUrls"));
        collect(task.getWorkflowInputs(), values);
        task.setInputMediaVersions(freeze(values, task.getInputMediaVersions(), actor));
    }

    public void validate(Map<String, Integer> versions, CanvasActor actor) {
        if (versions == null) {
            throw new GenerationJobException(409, "GENERATION_INPUT_SNAPSHOT_MISSING",
                    "此执行缺少输入媒体版本快照，请核实原任务后重新提交");
        }
        for (var entry : versions.entrySet()) {
            var resource = resources.get(entry.getKey(), actor);
            access.ready(resource, actor);
            if (entry.getValue() == null || resource.version() != entry.getValue()) {
                throw new GenerationJobException(409, "GENERATION_INPUT_CHANGED",
                        "已冻结的输入媒体版本发生变化，未继续提交供应商");
            }
        }
    }

    public void validateVideoMedia(VideoTask task, JSONObject config, CanvasActor actor) {
        if (!config.containsKey("videoRules")) {
            return;
        }
        List<String> images = new ArrayList<>();
        images.add(task.getFirstFrameImageUrl());
        images.add(task.getLastFrameImageUrl());
        images.addAll(references.parseJsonInputs(task.getReferenceImageUrls(), "referenceImageUrls"));
        VideoMediaRules.validate(config, task.getDuration(), facts(images, actor),
                facts(references.parseJsonInputs(task.getReferenceVideoUrls(), "referenceVideoUrls"), actor),
                facts(references.parseJsonInputs(task.getReferenceAudioUrls(), "referenceAudioUrls"), actor));
    }

    private List<MediaFacts> facts(List<String> values, CanvasActor actor) {
        return values.stream().filter(MediaReference::managed).map(value -> {
            var resource = resources.get(MediaReference.id(value), actor);
            access.ready(resource, actor);
            return resource.content();
        }).toList();
    }

    private Map<String, Integer> freeze(List<String> values, Map<String, Integer> expected, CanvasActor actor) {
        if (expected != null) {
            validate(expected, actor);
        }
        Map<String, Integer> versions = new LinkedHashMap<>(expected == null ? Map.of() : expected);
        for (String value : values) {
            if (!MediaReference.managed(value)) {
                continue;
            }
            var resource = resources.get(MediaReference.id(value), actor);
            access.ready(resource, actor);
            var frozenVersion = versions.putIfAbsent(resource.id(), resource.version());
            if (frozenVersion != null && frozenVersion != resource.version()) {
                throw new GenerationJobException(409, "GENERATION_INPUT_CHANGED",
                        "冻结输入期间媒体版本发生变化，请重新检查运行计划");
            }
        }
        return Map.copyOf(versions);
    }

    private void collect(JsonNode values, List<String> result) {
        if (values == null) {
            return;
        }
        if (values.isTextual()) {
            result.add(values.textValue());
        } else if (values.isContainerNode()) {
            values.forEach(value -> collect(value, result));
        }
    }
}
