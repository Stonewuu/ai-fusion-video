package com.stonewu.fusion.service.canvas.run;

import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.generation.VideoModelRules;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Component
@RequiredArgsConstructor
public class CanvasRunInputs {
    private final CanvasCapabilityRegistry capabilities;
    private final CanvasRunSources sources;

    public Ports resolve(NodePlan plan, Map<String, NodeRun> nodes, CanvasActor actor) {
        if (plan.action().equals("REUSE")) {
            return Ports.empty();
        }
        Map<String, List<Value>> result = new LinkedHashMap<>();
        for (var edge : plan.incoming()) {
            List<Value> values;
            if (plan.pinned().containsKey(edge.id())) {
                values = List.of(plan.pinned().get(edge.id()));
            } else {
                var upstream = nodes.get(edge.sourceNodeId());
                if (upstream == null || upstream.status() != State.SUCCEEDED) {
                    throw CanvasRunException.invalid("上游尚未成功，不能解析输入");
                }
                values = upstream.outputs().at(edge.sourcePort());
            }
            result.computeIfAbsent(edge.targetPort(), ignored -> new ArrayList<>()).addAll(values);
        }
        return configured(plan, result, actor);
    }

    public Ports configured(NodePlan plan, Map<String, List<Value>> bound, CanvasActor actor) {
        Map<String, List<Value>> result = new LinkedHashMap<>(bound);
        for (var port : capabilities.require(plan.node().kind()).inputs()) {
            var literal = plan.node().config().get(port.id());
            if (!result.containsKey(port.id()) && literal != null && !literal.isNull()) {
                if (!port.accepts(CanvasValueType.TEXT, plan.node().config())) {
                    throw CanvasRunException.invalid("此端口只能通过类型匹配的连接提供输入");
                }
                result.put(port.id(), List.of(sources.text(literal.asText())));
            }
        }
        if (capabilities.require(plan.node().kind()).inputs().stream().anyMatch(port -> port.id().equals("context"))) {
            CanvasCreativeInputs.compose(plan.node(), result, sources);
        }
        for (var port : capabilities.require(plan.node().kind()).inputs()) {
            var values = result.getOrDefault(port.id(), List.of());
            if (port.required() && !(port.id().equals("prompt") && optionalVideoPrompt(
                    plan)) && (values.isEmpty() || values.stream()
                    .anyMatch(value -> value.type() == CanvasValueType.TEXT
                            && (value.text() == null || value.text().isBlank())))) {
                throw CanvasRunException.invalid("输入“" + port.label() + "”为空");
            }
            if ((!port.collection() && values.size() > 1) || values.size() > 1000) {
                throw CanvasRunException.invalid("输入“" + port.label() + "”数量超过上限，请增加候选选择节点");
            }
            if (values.stream().anyMatch(value -> !port.accepts(value.type(), plan.node().config()))) {
                throw CanvasRunException.invalid("输入“" + port.label() + "”的实际类型不匹配");
            }
            values.forEach(value -> sources.authorize(value, actor, true));
        }
        return new Ports(result);
    }

    static boolean optionalVideoPrompt(NodePlan plan) {
        if (plan.node().kind() != CanvasNodeKind.VIDEO_GENERATION) {
            return false;
        }
        var config = plan.environment().path("model").path("config").asText();
        if (config.isBlank()) {
            return false;
        }
        var rules = VideoModelRules.from(JSONUtil.parseObj(config));
        return rules != null && Boolean.FALSE.equals(rules.promptRequired());
    }

    public Ports publish(NodeRun node, Ports output) {
        if (node.plan().action().equals("REUSE")) {
            return output;
        }
        Map<String, List<Value>> values = new LinkedHashMap<>();
        var capability = capabilities.require(node.plan().node().kind());
        for (var entry : output.values().entrySet()) {
            var port = capability.outputs().stream().filter(item -> item.id().equals(entry.getKey())).findFirst()
                    .orElseThrow(() -> CanvasRunException.invalid("执行器返回了未登记的输出端口"));
            if ((!port.collection() && entry.getValue().size() > 1) || entry.getValue().size() > 1000
                    || entry.getValue().stream()
                    .anyMatch(value -> value.type() != port.resolveType(node.plan().node().config()))) {
                throw CanvasRunException.invalid("执行器返回的输出类型或数量与协议不匹配");
            }
            List<Value> produced = new ArrayList<>();
            for (int index = 0; index < entry.getValue().size(); index++) {
                produced.add(sources.produced(entry.getValue().get(index), node.id(), port.id(), index));
            }
            values.put(entry.getKey(), List.copyOf(produced));
        }
        return new Ports(values);
    }
}
