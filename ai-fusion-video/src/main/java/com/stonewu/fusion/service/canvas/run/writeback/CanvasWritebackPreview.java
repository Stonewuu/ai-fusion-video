package com.stonewu.fusion.service.canvas.run.writeback;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.capability.CanvasWritebackFields;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasDocument;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.media.job.EpisodeComposeGuard;
import com.stonewu.fusion.service.resource.*;
import com.stonewu.fusion.service.resource.model.*;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;
import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Component
@RequiredArgsConstructor
public class CanvasWritebackPreview {
    private final CanvasRunSources sources;
    private final ResourceQueryService resources;
    private final ResourceAccessPolicy access;
    private final CanvasAccessPolicy canvasAccess;
    private final ProjectFields projects;
    private final ScriptFields scripts;
    private final StoryboardFields storyboards;
    private final AssetFields assets;
    private final CanvasJson json;
    private final EpisodeComposeGuard activeJobs;

    public void plan(CanvasDocument document, List<Value> targets, CanvasActor actor) {
        canvasAccess.requireEdit(document, actor);
        if (document.projectId() != null) {
            canvasAccess.requireProjectEditor(document.projectId(), actor);
        }
        for (var value : targets) {
            if (value.resource() == null) {
                throw CanvasRunException.invalid("写入目标必须是业务对象");
            }
            var ref = value.resource();
            var current = resources.get(ref.type(), ref.id(), actor);
            access.requireEdit(current.header(), actor);
            if (!Objects.equals(current.header().projectId(), document.projectId())) {
                throw CanvasRunException.invalid("目标业务对象不属于此画布的写入范围");
            }
            if (ref.version() == null || ref.version() != current.header().version()) {
                throw CanvasRunException.conflict("固定的写入目标已不是当前版本，请使用最新的目标对象");
            }
        }
    }

    public ObjectNode prepare(Run run, NodeRun node) {
        var field = CanvasWritebackFields.require(node.plan().node().config());
        if (!json.hash(field).equals(json.hash(node.plan().environment().path("writebackField")))) {
            throw CanvasRunException.conflict("写入字段协议已变化，请重新预览计划");
        }
        var target = target(node);
        var value = value(node);
        if (target.resource().type() != field.targetType() || value.type() != field.valueType()) {
            throw CanvasRunException.invalid("写入值或目标与字段类型不一致");
        }
        requireCurrent(run, node);
        validateText(field, value);
        JsonNode before = target.data();
        for (String segment : field.path().split("\\.")) {
            before = before.path(segment);
        }
        var preview = json.object(Map.of("field", field, "target", target.resource(), "value", value,
                "targetTitle", title(target), "before",
                before.isMissingNode() ? json.object(Map.of()).nullNode() : before));
        if (target.type() == ASSET_ITEM) {
            preview.put("expectedParentVersion", target.data().path("assetVersion").asInt());
            preview.put("notice", "同时校验并更新主资产版本；变体图片作为主封面时，主资产封面也会同步更新。");
        } else if (field.targetType() == SCRIPT && field.path().equals("rawContent")) {
            preview.put("dangerous", true);
            preview.put("notice",
                    "替换剧本原文会重置解析状态，清空原解析内容、故事梗概、角色信息、摘要、题材、受众和时长，并移除以下现有分集与场次。");
            var episodes = resources.scriptChildren(SCRIPT_EPISODE, target.resource().id(), run.actor());
            List<ObjectNode> affected = new ArrayList<>();
            for (var episode : episodes) {
                affected.add(json.object(Map.of("type", SCRIPT_EPISODE, "id", episode.path("id").asLong(), "title",
                        episode.path("title").asText(), "version", episode.path("version").asInt())));
                for (var scene : resources.scriptChildren(SCRIPT_SCENE, episode.path("id").asLong(), run.actor())) {
                    affected.add(json.object(Map.of("type", SCRIPT_SCENE, "id", scene.path("id").asLong(), "title",
                            scene.path("sceneHeading").asText(), "version", scene.path("version").asInt())));
                }
            }
            preview.set("removedResources", json.object(Map.of("items", affected)).path("items"));
        } else if (field.targetType() == STORYBOARD_EPISODE && field.valueType() != TEXT) {
            preview.put("notice", field.valueType() == VIDEO
                    ? "只有原分集合成已结束时才能采用；采用成片会更新成片状态和时间，并解除旧合成任务绑定。"
                    : "只有原分集合成已结束时才能采用；采用音频会解除旧配乐任务绑定并清理原音频错误状态。");
        }
        if (json.write(preview).length() > 4 * 1024 * 1024) {
            throw CanvasRunException.invalid("写入预览超过 4 MiB，请拆分内容或在原页面处理");
        }
        preview.put("hash", json.hash(Map.of("preview", preview.deepCopy(), "nodeRunId", node.id(), "fingerprint",
                node.plan().fingerprint())));
        return preview;
    }

    public ResourceSnapshot requireCurrent(Run run, NodeRun node) {
        var target = target(node);
        var ref = target.resource();
        var current = resources.get(ref.type(), ref.id(), run.actor());
        access.requireEdit(current.header(), run.actor());
        if (!Objects.equals(current.header().projectId(), run.plan().graph().projectId())) {
            throw CanvasRunException.invalid("目标业务对象不属于此画布的写入范围");
        }
        if (ref.version() == null || ref.version() != current.header().version()) {
            throw CanvasRunException.conflict(
                    "目标内容已修改，不能覆盖新版本；请跳过本次写入，再复用生成结果重新制定计划");
        }
        if (ref.type() == ASSET_ITEM && (!target.data().has("assetVersion")
                || target.data().path("assetVersion").asInt() != current.data().path("assetVersion").asInt())) {
            throw CanvasRunException.conflict("主资产或其他变体已修改，请重新确认目标版本");
        }
        if (ref.type() == STORYBOARD_EPISODE && value(node).type() != TEXT) {
            activeJobs.requireIdle(ref.id(), current.data());
        }
        node.inputs().all().forEach(input -> sources.authorize(input, run.actor(), true));
        return current;
    }

    public ResourceCommand command(NodeRun node) {
        var field = CanvasWritebackFields.require(node.plan().node().config());
        var target = target(node);
        var source = value(node);
        var ref = target.resource();
        String requestId = "canvas-writeback:" + node.id();
        Integer parent = ref.type() == ASSET_ITEM ? target.data().path("assetVersion").asInt() : null;
        int version = Math.toIntExact(ref.version());
        if (source.type() == TEXT) {
            if (ref.type() == SCRIPT && field.path().equals("rawContent")) {
                return new ScriptSourceCommand(requestId, ref.id(), version, source.text());
            }
            return new ResourcePatchCommand(requestId, ref.type(), ref.id(), version,
                    json.object(Map.of(field.path(), source.text())), parent);
        }
        if (ref.type() == STORYBOARD_EPISODE) {
            return new AdoptEpisodeMediaCommand(requestId, ref.id(), version, field.path(), source.mediaId());
        }
        return new ResourceBindMediaCommand(requestId, ref.type(), ref.id(), version, parent,
                Map.of(field.path(), source.mediaId()));
    }

    private void validateText(CanvasWritebackFields.Field field, Value value) {
        if (value.type() != TEXT) {
            if (value.mediaId() == null || value.mediaVersion() == null) {
                throw CanvasRunException.invalid("写入前必须完成媒体登记");
            }
            return;
        }
        if (value.text() == null || value.text().isBlank()) {
            throw CanvasRunException.invalid("不能写入空的生成文本");
        }
        var patch = json.object(Map.of(field.path(), value.text()));
        switch (field.targetType()) {
            case PROJECT -> projects.values(patch, false);
            case SCRIPT, SCRIPT_EPISODE, SCRIPT_SCENE -> scripts.values(field.targetType(), patch);
            case STORYBOARD, STORYBOARD_EPISODE, STORYBOARD_SCENE, SHOT ->
                    storyboards.values(field.targetType(), patch, false);
            case ASSET, ASSET_ITEM -> assets.values(field.targetType(), patch, false);
            default -> throw CanvasRunException.invalid("此对象不接受文本写入");
        }
    }

    private Value target(NodeRun node) {
        if (node.inputs().at("target").size() != 1 || node.inputs().at("target").getFirst().resource() == null) {
            throw CanvasRunException.invalid("请连接一个具体业务对象作为写入目标");
        }
        return node.inputs().at("target").getFirst();
    }

    private Value value(NodeRun node) {
        if (node.inputs().at("value").size() != 1) {
            throw CanvasRunException.invalid("请先选择一个具体结果再写入业务内容");
        }
        return node.inputs().at("value").getFirst();
    }

    private String title(Value target) {
        String field = switch (target.type()) {
            case PROJECT, ASSET, ASSET_ITEM -> "name";
            case SCRIPT_EPISODE, STORYBOARD_EPISODE -> "title";
            case SCRIPT_SCENE, STORYBOARD_SCENE -> "sceneHeading";
            case SHOT -> "shotNumber";
            default -> "title";
        };
        String title = target.data().path(field).asText("");
        return title.isBlank() ? "编号 " + target.resource().id() : title;
    }
}
