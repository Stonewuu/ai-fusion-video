package com.stonewu.fusion.service.canvas.run.mcp;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.ExternalOperation;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodePlan;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Value;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunSources;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.Contract;
import static com.stonewu.fusion.service.canvas.run.mcp.CanvasMcpData.Idempotency;

@Component
@RequiredArgsConstructor
public class CanvasMcpPlanner {
    private final CanvasJson json;
    private final CanvasMcpProcessor processor;
    private final CanvasRunSources sources;

    public void validate(CanvasNode node, NodePlan plan, Map<String, NodePlan> nodes, boolean batch) {
        var environment = batch ? plan.environment().path("operation") : plan.environment();
        var contract = json.read(environment.path("mcpContract").toString(), Contract.class);
        List<Value> values = new ArrayList<>();
        boolean dynamic = batch;
        for (var edge : plan.incoming()) {
            if (edge.targetPort().equals("input")) {
                if (plan.pinned().containsKey(edge.id())) {
                    values.add(plan.pinned().get(edge.id()));
                } else if (nodes.containsKey(edge.sourceNodeId())) {
                    var upstream = nodes.get(edge.sourceNodeId());
                    var known = upstream.sourceOutputs().at(edge.sourcePort());
                    values.addAll(known);
                    if (known.isEmpty() && !upstream.action().equals("REUSE")) {
                        dynamic = true;
                    }
                }
            }
        }
        if (values.isEmpty() && !dynamic && node.config().hasNonNull("input")) {
            values.add(sources.text(node.config().path("input").asText()));
        }
        validateInputs(node, contract, values, dynamic);
    }

    /**
     * 运行计划与模板重绑共用参数检查；动态输入只延后正文校验，字段与幂等约束仍立即检查。
     */
    public void validateInputs(CanvasNode node, Contract contract, List<Value> values, boolean dynamic) {
        String mode = node.config().path("inputMode").asText("NONE");
        if (!dynamic) {
            processor.arguments(node, contract, new Ports(values.isEmpty() ? Map.of() : Map.of("input", values)),
                    "00000000-0000-0000-0000-000000000000");
        } else {
            if (mode.equals("NONE")) {
                throw CanvasRunException.invalid("MCP 已连接动态输入，请指定传给工具的方式");
            }
            if (mode.equals("TEXT_FIELD") && !contract.definition().tool().inputSchema().path("properties")
                    .path(node.config().path("inputParameter").asText()).path("type").asText().equals("string")) {
                throw CanvasRunException.invalid("MCP 文本输入必须绑定到工具声明的字符串参数");
            }
            var args = json.read(node.config().path("argumentsJson").asText("{}"), Map.class);
            if (args == null || mode.equals("JSON_OBJECT") && !args.isEmpty()) {
                throw CanvasRunException.invalid("上游完整 JSON 参数不能与固定参数混合");
            }
            if (contract.definition().idempotency() == Idempotency.KEYED && args.containsKey(
                    contract.definition().idempotencyParameter())) {
                throw CanvasRunException.invalid("MCP 幂等键由系统填入，不能配置固定值");
            }
        }
    }

    public ExternalOperation describe(NodePlan plan) {
        if (plan.action().equals("REUSE")) {
            return null;
        }
        boolean batch = plan.node().kind() == CanvasNodeKind.FOREACH;
        var environment = batch ? plan.environment().path("operation") : plan.environment();
        if (!environment.has("mcpContract")) {
            return null;
        }
        var contract = json.read(environment.path("mcpContract").toString(), Contract.class);
        var definition = contract.definition();
        return new ExternalOperation(contract.name(), definition.tool().serverName(), definition.tool().toolName(),
                definition.effect().name(),
                definition.idempotency().name(), definition.timeoutSeconds(),
                batch ? plan.node().config().path("maxItems").asInt() : 1);
    }
}
