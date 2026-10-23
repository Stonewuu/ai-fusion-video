package com.stonewu.fusion.service.canvas.run;

import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.entity.generation.ImageItem;
import com.stonewu.fusion.entity.generation.VideoItem;
import com.stonewu.fusion.mapper.generation.ImageItemMapper;
import com.stonewu.fusion.mapper.generation.VideoItemMapper;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.job.GenerationJobCommands;
import com.stonewu.fusion.service.generation.job.GenerationJobRepository;
import com.stonewu.fusion.service.generation.job.GenerationSubmission;
import com.stonewu.fusion.service.generation.job.GenerationSubmissionService;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;
import static com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Executor;
import static com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Outcome;

@Component
@RequiredArgsConstructor
public class CanvasGenerationExecutor implements Executor {
    private final GenerationSubmissionService submissions;
    private final GenerationJobRepository jobs;
    private final GenerationJobCommands commands;
    private final CanvasRunSources sources;
    private final CanvasJson json;
    private final ImageItemMapper imageItems;
    private final VideoItemMapper videoItems;

    @Override
    public Set<CanvasNodeKind> kinds() {
        return CanvasRunEnvironment.GENERATION;
    }

    @Override
    public Outcome step(Run run, NodeRun node, Claim claim) {
        if (node.externalId() == null) {
            GenerationSubmission submission = submit(run, node);
            return new Outcome(State.RUNNING, Ports.empty(), "GENERATION", submission.id(), null);
        }
        var submission = jobs.submission(node.externalId());
        if (!node.id().equals(submission.canvasNodeRunId())) {
            throw CanvasRunException.invalid("生成任务与节点运行关联不一致");
        }
        State state = switch (submission.state()) {
            case "SUCCEEDED" -> State.SUCCEEDED;
            case "PARTIAL_FAILED" -> State.PARTIAL_FAILED;
            case "FAILED" -> State.FAILED;
            case "CANCELLED" -> State.CANCELLED;
            case "UNKNOWN_SUBMISSION" -> State.UNKNOWN_SUBMISSION;
            case "NEEDS_REVIEW" -> State.NEEDS_REVIEW;
            case "CANCEL_REQUESTED" -> State.CANCEL_REQUESTED;
            default -> State.RUNNING;
        };
        String error = jobs.jobs(submission.id()).stream().filter(item -> item.errorMessage() != null)
                .map(item -> item.errorMessage()).findFirst().orElse(null);
        Ports outputs = outputs(run, node, submission, state == State.SUCCEEDED || state == State.PARTIAL_FAILED);
        return new Outcome(state, outputs, "GENERATION", submission.id(), error);
    }

    @Override
    public void cancel(Run run, NodeRun node) {
        if (node.externalId() == null) {
            return;
        }
        var submission = jobs.submission(node.externalId());
        commands.cancel(submission.kind(), submission.taskId(), run.actor());
    }

    private GenerationSubmission submit(Run run, NodeRun node) {
        var config = node.plan().node().config();
        var frozen = json.read(node.plan().environment().path("model").toString(), AiModel.class);
        String apiHash = node.plan().environment().path("apiHash").asText();
        String key = "canvas-node:" + node.id();
        // 父运行由节点租约保护；生成工作者使用自己的身份和租约，无需保持原浏览器或 Agent 工具在线。
        var context = ResourceCommandContext.user(run.actor(), run.canvasId());
        String prompt = node.inputs().at("prompt").isEmpty() ? "" : node.inputs().at("prompt").getFirst().text();
        var inputVersions = node.inputs().all().stream().filter(value -> value.mediaId() != null)
                .collect(Collectors.toMap(Value::mediaId, Value::mediaVersion, (first, second) -> first));
        if (CanvasRunEnvironment.kind(node.plan().node()) == GenerationKind.IMAGE) {
            var task = ImageTask.builder().projectId(run.plan().graph().projectId()).modelId(frozen.getId())
                    .workflowVersionId(number(config, "workflowVersionId")).prompt(prompt)
                    .count(config.path("count").asInt(1))
                    .refImageUrls(mediaList(node.inputs(), "referenceImages")).ratio(string(config, "ratio"))
                    .resolution(string(config, "resolution")).width(integer(config, "width"))
                    .height(integer(config, "height"))
                    .category("canvas").build();
            task.setWorkflowInputs(workflow(node));
            task.setInputMediaVersions(inputVersions);
            return submissions.imageFrozen(task, key, context, node.id(), frozen, apiHash);
        }
        int defaultDuration = JSONUtil.parseObj(frozen.getConfig()).getInt("defaultDuration",
                node.plan().videoSeconds() / config.path("count").asInt(1));
        int duration = config.path("duration").asInt(defaultDuration);
        String firstFrame = mediaOne(node.inputs(), "firstFrame");
        var task = VideoTask.builder().projectId(run.plan().graph().projectId()).modelId(frozen.getId())
                .workflowVersionId(number(config, "workflowVersionId")).prompt(prompt)
                .count(config.path("count").asInt(1))
                .firstFrameImageUrl(firstFrame).lastFrameImageUrl(mediaOne(node.inputs(), "lastFrame"))
                .referenceImageUrls(mediaList(node.inputs(), "referenceImages"))
                .referenceVideoUrls(mediaList(node.inputs(), "referenceVideos"))
                .referenceAudioUrls(mediaList(node.inputs(), "referenceAudios"))
                .generateMode(firstFrame == null ? "text2video" : "image2video")
                .ratio(string(config, "ratio")).resolution(string(config, "resolution")).duration(duration)
                .seed(number(config, "seed"))
                .watermark(flag(config, "watermark")).generateAudio(flag(config, "generateAudio"))
                .cameraFixed(flag(config, "cameraFixed"))
                .category("canvas").build();
        task.setWorkflowInputs(workflow(node));
        task.setInputMediaVersions(inputVersions);
        return submissions.videoFrozen(task, key, context, node.id(), frozen, apiHash);
    }

    private ObjectNode workflow(NodeRun node) {
        if (node.plan().node().kind() != CanvasNodeKind.COMFYUI) {
            return null;
        }
        var value = node.plan().node().config().deepCopy();
        value.remove(List.of("modelId", "workflowVersionId", "resultType"));
        node.inputs().values().forEach((port, entries) -> {
            if (port.equals("prompt")) {
                value.put(port, entries.getFirst().text());
            } else if (port.equals("firstFrame") || port.equals("lastFrame")) {
                if (!entries.isEmpty()) {
                    value.put(port, MediaReference.of(entries.getFirst().mediaId()));
                }
            } else if (!entries.isEmpty()) {
                var values = value.putArray(port);
                entries.forEach(entry -> values.add(MediaReference.of(entry.mediaId())));
            }
        });
        return value;
    }

    private Ports outputs(Run run, NodeRun node, GenerationSubmission submission, boolean requireResult) {
        Map<String, List<Value>> output = new LinkedHashMap<>();
        boolean comfy = node.plan().node().kind() == CanvasNodeKind.COMFYUI;
        if (submission.kind() == GenerationKind.IMAGE) {
            var values = imageItems.selectList(new LambdaQueryWrapper<ImageItem>()
                    .select(ImageItem::getImageUrl).eq(ImageItem::getTaskId, submission.taskId())
                    .eq(ImageItem::getStatus, 1).orderByAsc(ImageItem::getId))
                    .stream().map(item -> output(run, item.getImageUrl())).toList();
            output.put(comfy ? "results" : "images", values);
        } else {
            var rows = videoItems.selectList(new LambdaQueryWrapper<VideoItem>()
                    .select(VideoItem::getVideoUrl, VideoItem::getFirstFrameUrl, VideoItem::getLastFrameUrl)
                    .eq(VideoItem::getTaskId, submission.taskId()).eq(VideoItem::getStatus, 1).orderByAsc(VideoItem::getId));
            var videos = new ArrayList<Value>();
            var first = new ArrayList<Value>();
            var last = new ArrayList<Value>();
            for (var row : rows) {
                videos.add(output(run, row.getVideoUrl()));
                if (row.getFirstFrameUrl() != null) {
                    first.add(output(run, row.getFirstFrameUrl()));
                }
                if (row.getLastFrameUrl() != null) {
                    last.add(output(run, row.getLastFrameUrl()));
                }
            }
            output.put(comfy ? "results" : "videos", videos);
            if (!comfy) {
                output.put("firstFrames", first);
                output.put("lastFrames", last);
            }
        }
        if (requireResult && output.values().stream().allMatch(List::isEmpty)) {
            throw CanvasRunException.invalid("生成任务没有已登记的输出，请核实原任务");
        }
        return new Ports(output);
    }

    private Value output(Run run, String reference) {
        if (!MediaReference.managed(reference)) {
            throw CanvasRunException.invalid("生成结果尚未登记为托管媒体");
        }
        return sources.mediaValue(MediaReference.id(reference), run.actor(), run.plan().graph().projectId());
    }

    private String mediaList(Ports ports, String port) {
        return ports.at(port).isEmpty() ? null : json.write(
                ports.at(port).stream().map(value -> MediaReference.of(value.mediaId())).toList());
    }

    private String mediaOne(Ports ports, String port) {
        return ports.at(port).isEmpty() ? null : MediaReference.of(ports.at(port).getFirst().mediaId());
    }

    private String string(ObjectNode config, String field) {
        return config.has(field) ? config.path(field).asText() : null;
    }

    private Integer integer(ObjectNode config, String field) {
        return config.has(field) ? config.path(field).asInt() : null;
    }

    private Long number(ObjectNode config, String field) {
        return config.has(field) ? config.path(field).asLong() : null;
    }

    private Boolean flag(ObjectNode config, String field) {
        return config.has(field) ? config.path(field).asBoolean() : null;
    }
}
