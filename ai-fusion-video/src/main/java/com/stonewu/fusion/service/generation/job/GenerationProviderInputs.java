package com.stonewu.fusion.service.generation.job;

import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.media.MediaInputTransport;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.model.MediaKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * 每次使用固定输入重新签发传输地址，不把短时地址写回任务或画布。
 */
@Component
@RequiredArgsConstructor
public class GenerationProviderInputs {
    private final CanvasJson json;
    private final GenerationModelCapabilityService capabilities;
    private final ReferenceImageTransportService references;
    private final MediaInputTransport media;
    private final GenerationInputVersions versions;

    public ImageTask image(RemoteGenerationJob job, GenerationSubmission submission, ApiConfig api,
                           boolean resolveMedia) {
        var body = job.input().deepCopy();
        body.remove("_resultItemIds");
        var task = json.read(json.write(body), ImageTask.class);
        task.setExecutionContextJson(json.write(GenerationExecutionIdentity.worker(submission.context())));
        if (resolveMedia) {
            versions.validate(task.getInputMediaVersions(), submission.context().actor());
            capabilities.validateImageTask(submission.model(), task);
            var values = references.parseJsonInputs(task.getRefImageUrls(), "refImageUrls");
            if (!values.isEmpty()) {
                task.setRefImageUrls(JSONUtil.toJsonStr(references.resolveInputs(submission.model(),
                        capabilities.getMergedModelConfig(submission.model()), values, api,
                        submission.context().actor(),
                        job.claim())));
            }
            task.setWorkflowInputs(workflow(task.getWorkflowInputs(), submission, api, job.claim()));
        }
        return task;
    }

    public VideoTask video(RemoteGenerationJob job, GenerationSubmission submission, ApiConfig api,
                           boolean resolveMedia) {
        var body = job.input().deepCopy();
        body.remove("_resultItemIds");
        var task = json.read(json.write(body), VideoTask.class);
        task.setExecutionContextJson(json.write(GenerationExecutionIdentity.worker(submission.context())));
        if (!resolveMedia) {
            return task;
        }
        versions.validate(task.getInputMediaVersions(), submission.context().actor());
        capabilities.validateVideoTask(submission.model(), task);
        versions.validateVideoMedia(task, capabilities.getMergedModelConfig(submission.model()),
                submission.context().actor());
        if (task.getFirstFrameImageUrl() != null) {
            task.setFirstFrameImageUrl(
                    images(List.of(task.getFirstFrameImageUrl()), submission, api, job.claim()).getFirst());
        }
        if (task.getLastFrameImageUrl() != null) {
            task.setLastFrameImageUrl(
                    images(List.of(task.getLastFrameImageUrl()), submission, api, job.claim()).getFirst());
        }
        var refs = references.parseJsonInputs(task.getReferenceImageUrls(), "referenceImageUrls");
        if (!refs.isEmpty()) {
            task.setReferenceImageUrls(JSONUtil.toJsonStr(images(refs, submission, api, job.claim())));
        }
        task.setReferenceVideoUrls(
                mediaList(task.getReferenceVideoUrls(), "referenceVideoUrls", MediaKind.VIDEO, submission,
                        job.claim()));
        task.setReferenceAudioUrls(
                mediaList(task.getReferenceAudioUrls(), "referenceAudioUrls", MediaKind.AUDIO, submission,
                        job.claim()));
        task.setWorkflowInputs(workflow(task.getWorkflowInputs(), submission, api, job.claim()));
        return task;
    }

    private List<String> images(List<String> input, GenerationSubmission submission, ApiConfig api,
                                RemoteGenerationJob.Claim claim) {
        return references.resolveInputs(submission.model(), capabilities.getMergedModelConfig(submission.model()),
                input, api, submission.context().actor(), claim);
    }

    private ObjectNode workflow(ObjectNode raw, GenerationSubmission submission, ApiConfig api,
                                RemoteGenerationJob.Claim claim) {
        if (raw == null) {
            return null;
        }
        ObjectNode result = raw.deepCopy();
        for (String port : List.of("firstFrame", "lastFrame")) {
            if (raw.hasNonNull(port)) {
                result.put(port, images(List.of(raw.path(port).asText()), submission, api, claim).getFirst());
            }
        }
        for (String port : List.of("referenceImages", "referenceVideos", "referenceAudios")) {
            if (!raw.has(port)) {
                continue;
            }
            List<String> values = new ArrayList<>();
            raw.path(port).forEach(value -> values.add(value.asText()));
            List<String> resolved = port.equals("referenceImages") ? images(values, submission, api, claim)
                    : values.stream().map(value -> MediaReference.managed(value) ? media.generationUrl(value,
                    submission.context().actor(),
                    port.equals("referenceVideos") ? MediaKind.VIDEO : MediaKind.AUDIO, claim) : value).toList();
            var output = result.putArray(port);
            resolved.forEach(output::add);
        }
        return result;
    }

    private String mediaList(String input, String field, MediaKind kind, GenerationSubmission submission,
                             RemoteGenerationJob.Claim claim) {
        if (input == null || input.isBlank()) {
            return input;
        }
        return JSONUtil.toJsonStr(references.parseJsonInputs(input, field).stream().map(value ->
                MediaReference.managed(value) ? media.generationUrl(value, submission.context().actor(), kind,
                        claim) : value).toList());
    }
}
