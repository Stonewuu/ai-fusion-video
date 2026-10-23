package com.stonewu.fusion.service.canvas.agent.node;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.controller.ai.vo.AiMultimodalInputVO;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelMultimodalCapabilities;
import com.stonewu.fusion.service.ai.agentscope.message.AgentScopeMessageMapper;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodePlan;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import io.agentscope.core.message.Msg;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.*;

@Component
@RequiredArgsConstructor
public class CanvasAgentNodeInputs {
    private final CanvasRunSources sources;
    private final MediaResourceQueries media;
    private final AgentScopeMessageMapper messages;
    private final CanvasJson json;
    private final CanvasCapabilityRegistry capabilities;

    public void validate(String type, NodeRun node) {
        require(CanvasAgentNodeContracts.require(type), node.inputs().all().stream().map(Value::type).toList(), false);
    }

    private void require(CanvasAgentNodeContracts.Profile profile, List<CanvasValueType> types, boolean batch) {
        long count = types.stream().filter(profile.inputTypes()::contains).count();
        if (count == 0 || !batch && profile.singleInput() && count != 1) {
            throw CanvasRunException.invalid("专业智能体输入不完整或数量不符。" + profile.hint());
        }
    }

    /**
     * 已知资源在预览时检查；动态上游按端口契约检查，执行时仍重验真实数量。
     */
    public void plan(CanvasNode node, NodePlan plan, Map<String, NodePlan> upstream, boolean batch) {
        var profile = CanvasAgentNodeContracts.require(CanvasAgentNodeTypes.type(node));
        List<CanvasValueType> types = new ArrayList<>();
        List<Value> known = new ArrayList<>();
        for (var edge : plan.incoming()) {
            if (plan.pinned().containsKey(edge.id())) {
                var value = plan.pinned().get(edge.id());
                types.add(value.type());
                known.add(value);
            } else if (upstream.containsKey(edge.sourceNodeId())) {
                var source = upstream.get(edge.sourceNodeId());
                var values = source.sourceOutputs().at(edge.sourcePort());
                if (!values.isEmpty()) {
                    values.forEach(value -> types.add(value.type()));
                    known.addAll(values);
                } else if (source.sourceOutputs().values().isEmpty() && !source.action().equals("REUSE")) {
                    capabilities.require(source.node().kind()).outputs().stream()
                            .filter(port -> port.id().equals(edge.sourcePort()))
                            .findFirst().ifPresent(port -> types.add(port.resolveType(source.node().config())));
                }
            }
        }
        for (var port : capabilities.require(node.kind()).inputs()) {
            if (port.accepts(CanvasValueType.TEXT, node.config()) && !node.config().path(port.id()).asText("")
                    .isBlank()) {
                types.add(CanvasValueType.TEXT);
            }
        }
        require(profile, types, batch);
        var environment = batch ? plan.environment().path("operation") : plan.environment();
        var model = json.read(environment.path("model").toString(), AiModel.class);
        for (var type : types) {
            String mediaType = switch (type) {
                case IMAGE -> "image";
                case VIDEO -> "video";
                case AUDIO -> "audio";
                case DOCUMENT -> "file";
                default -> null;
            };
            if (mediaType != null) {
                AiModelMultimodalCapabilities.managedTypeTransport(model, mediaType);
            }
        }
        if (known.stream().map(Value::mediaId).filter(id -> id != null).distinct().count() > 32) {
            throw CanvasRunException.invalid("单次智能体最多观察 32 个媒体输入，请按批次处理");
        }
        for (var value : known) {
            if (value.mediaId() != null) {
                var mime = value.data().path("content").path("mimeType").asText(null);
                if (mime != null) {
                    var transport = AiModelMultimodalCapabilities.managedTransport(model, mime);
                    AiModelMultimodalCapabilities.validateUpload(model, mime, transport);
                }
            }
        }
    }

    /**
     * 只把唯一的明确输入或共同祖先绑定到单对象模板，集合输入不随 Map 顺序随机选第一项。
     */
    public void context(NodeRun node, CanvasAgentNodeScope scope, Map<String, Object> variables) {
        bind(node, scope, variables, CanvasValueType.SCRIPT, "scriptId");
        bind(node, scope, variables, CanvasValueType.SCRIPT_EPISODE, "scriptEpisodeId");
        if (variables.containsKey("scriptEpisodeId")) {
            variables.put("episodeId", variables.get("scriptEpisodeId"));
        }
        bind(node, scope, variables, CanvasValueType.STORYBOARD, "storyboardId");
        bind(node, scope, variables, CanvasValueType.STORYBOARD_EPISODE, "storyboardEpisodeId");
        bind(node, scope, variables, CanvasValueType.SHOT, "storyboardItemId");
        bind(node, scope, variables, CanvasValueType.ASSET_ITEM, "assetItemId");
    }

    private void bind(NodeRun node, CanvasAgentNodeScope scope, Map<String, Object> variables, CanvasValueType type,
                      String name) {
        var direct = node.inputs().all().stream()
                .filter(value -> value.resource() != null && value.resource().type() == type)
                .map(value -> value.resource().id()).distinct().toList();
        var candidates = direct.isEmpty() ? scope.resources().values().stream().filter(entry -> entry.type() == type)
                .map(CanvasAgentNodeScope.Entry::id).distinct().toList() : direct;
        if (candidates.size() == 1) {
            variables.put(name, candidates.getFirst());
        }
    }

    public List<Msg> messages(Run run, NodeRun node, ObjectNode input, AiModel model) {
        List<AiMultimodalInputVO> attachments = new ArrayList<>();
        Set<String> included = new LinkedHashSet<>();
        for (var value : node.inputs().all()) {
            if (value.mediaId() != null) {
                sources.authorize(value, run.actor(), true);
                if (!included.add(value.mediaId())) {
                    continue;
                }
                var resource = media.get(value.mediaId(), run.actor());
                if (resource.content().byteSize() > 64L * 1024 * 1024) {
                    throw CanvasRunException.invalid("智能体单个输入文件超过 64 MiB，请先截取所需内容");
                }
                String transport = AiModelMultimodalCapabilities.managedTransport(model, resource.content().mimeType());
                String type = AiModelMultimodalCapabilities.validateUpload(model, resource.content().mimeType(),
                        transport);
                var attachment = new AiMultimodalInputVO();
                attachment.setId(resource.id());
                attachment.setName(resource.name());
                attachment.setMimeType(resource.content().mimeType());
                attachment.setSize(resource.content().byteSize());
                attachment.setInputType(type);
                // 保存稳定托管引用，模型边界按所选模型能力转为私有 Base64 或有时限的授权票据。
                attachment.setTransport("url");
                attachment.setUrl(MediaReference.of(resource.id()));
                attachments.add(attachment);
            }
        }
        if (attachments.size() > 32) {
            throw CanvasRunException.invalid("单次智能体最多观察 32 个媒体输入，请按批次处理");
        }
        String instruction = json.write(input);
        if (!attachments.isEmpty()) {
            instruction += "\n附带的是所选媒体原文件。视频采样和文档读取范围取决于模型能力，请明确观察范围，不能仅根据文件名或地址声称已经检查全部内容。";
        }
        return messages.toUserMessages(instruction, attachments);
    }
}
