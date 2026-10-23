package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Run;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import lombok.RequiredArgsConstructor;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.entity.ai.AgentMessage;
import com.stonewu.fusion.mapper.ai.AgentMessageMapper;
import com.stonewu.fusion.mapper.canvas.CanvasGenerationUsageMapper;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

@Component
@RequiredArgsConstructor
public class CanvasAgentNodeOutputs {
    private final CanvasCapabilityRegistry capabilities;
    private final CanvasRunSources sources;
    private final ResourceQueryService resources;
    private final AgentMessageMapper messages;
    private final CanvasGenerationUsageMapper usage;

    public Ports collect(Run run, NodeRun node, CanvasAgentNodeRepository.Job job) {
        var output = new LinkedHashMap<String, List<Value>>();
        for (var port : capabilities.require(node.plan().node().kind()).outputs()) {
            CanvasValueType type = port.resolveType(node.plan().node().config());
            List<Value> values = new ArrayList<>();
            if (type == CanvasValueType.TEXT) {
                var history = messages.selectList(
                        new LambdaQueryWrapper<AgentMessage>().select(AgentMessage::getContent)
                                .eq(AgentMessage::getRunId, job.agentRunId()).eq(AgentMessage::getRole, "assistant")
                                .isNull(AgentMessage::getParentToolCallId).isNotNull(AgentMessage::getContent)
                                .orderByAsc(AgentMessage::getMessageOrder));
                if (!history.isEmpty()) {
                    values.add(sources.text(history.getLast().getContent()));
                }
            } else if (type == CanvasValueType.IMAGE || type == CanvasValueType.VIDEO) {
                values.addAll(mediaValues(run, node, type));
            } else {
                for (var entry : job.scope().resources().values()) {
                    if (entry.type() == type && entry.changed() && !entry.deleted()) {
                        var value = resources.get(type, entry.id(), run.actor());
                        if (value.header().version() != entry.version()) {
                            throw CanvasRunException.conflict(
                                    "专业智能体完成后业务结果已变化，不能发布新的内容冒充原输出");
                        }
                        values.add(sources.business(value));
                    }
                }
            }
            if (values.isEmpty() && !port.id().equals("summary")) {
                throw CanvasRunException.invalid(
                        "智能体没有提交符合“" + port.label() + "”类型的真实结果，请查看任务记录");
            }
            if (!port.collection() && values.size() > 1) {
                throw CanvasRunException.invalid("智能体返回多个业务结果，但此输出仅接收一个对象");
            }
            output.put(port.id(), List.copyOf(values));
        }
        return new Ports(output);
    }

    /**
     * Publish ready media for inspection while other work is still running; dependencies still wait for node success.
     */
    public Ports preview(Run run, NodeRun node) {
        var output = new LinkedHashMap<String, List<Value>>();
        for (var port : capabilities.require(node.plan().node().kind()).outputs()) {
            CanvasValueType type = port.resolveType(node.plan().node().config());
            if (type == CanvasValueType.IMAGE || type == CanvasValueType.VIDEO) {
                output.put(port.id(), mediaValues(run, node, type));
            }
        }
        return new Ports(output);
    }

    private List<Value> mediaValues(Run run, NodeRun node, CanvasValueType type) {
        var references = type == CanvasValueType.IMAGE ? usage.successfulImages(node.id()) : usage.successfulVideos(
                node.id());
        List<Value> values = new ArrayList<>();
        for (String reference : references) {
            if (!MediaReference.managed(reference)) {
                throw CanvasRunException.invalid("智能体生成结果尚未登记为托管媒体");
            }
            values.add(sources.mediaValue(MediaReference.id(reference), run.actor(), run.plan().graph().projectId()));
        }
        return List.copyOf(values);
    }
}
