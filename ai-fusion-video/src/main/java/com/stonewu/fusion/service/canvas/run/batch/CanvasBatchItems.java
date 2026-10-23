package com.stonewu.fusion.service.canvas.run.batch;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.run.CanvasRunEnvironment;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import com.stonewu.fusion.service.canvas.run.CanvasRunInputs;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.canvas.CanvasNodeRunMapper;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

/**
 * 父运行锁内原子冻结所有子项；恢复只观察原子项，重试必须是显式的新 attempt。
 */
@Component
@RequiredArgsConstructor
public class CanvasBatchItems {
    private final CanvasRunRepository runs;
    private final CanvasRunEnvironment environments;
    private final CanvasRunInputs inputs;
    private final CanvasJson json;
    private final CanvasNodeRunMapper nodeRuns;

    @Transactional(propagation = Propagation.MANDATORY)
    public void expand(Run run, NodeRun parent) {
        var values = parent.inputs().at("items");
        if (values.isEmpty() || values.size() > parent.plan().node().config().path("maxItems").asInt()
                || values.size() > run.plan().budget().maxBatchItems()) {
            throw CanvasRunException.invalid("实际批量数量超出此次确认的范围，请减少输入或重新预览");
        }
        var body = environments.batchBody(parent.plan().node());
        var environment = (ObjectNode) parent.plan().environment().path("operation");
        var plan = new NodePlan(body,
                json.hash(Map.of("parent", parent.plan().fingerprint(), "operation", body.config())),
                "EXECUTE", null, List.of(), List.of(), Map.of(), Ports.empty(), environment,
                environments.images(body), environments.seconds(body, environment));
        String port = parent.plan().node().config().path("inputPort").asText();
        var targets = parent.checkpoint().putArray("targetIds");
        for (int index = 0; index < values.size(); index++) {
            var value = values.get(index);
            var resolved = inputs.configured(plan, Map.of(port, List.of(value)), run.actor());
            insert(parent, plan, resolved, index, 1);
            targets.add(value.id());
        }
        parent.checkpoint().put("targetHash", json.hash(values)).put("itemCount", values.size());
    }

    @Transactional(propagation = Propagation.MANDATORY)
    public void retry(NodeRun parent, List<NodeRun> failed) {
        for (var item : failed) {
            insert(parent, item.plan(), item.inputs(), item.itemIndex(), item.attempt() + 1);
        }
    }

    private void insert(NodeRun parent, NodePlan plan, Ports input, int index, int attempt) {
        nodeRuns.insertBatchItem(UUID.randomUUID().toString(), index, attempt, plan.fingerprint(), json.write(plan),
                json.write(input), json.write(Ports.empty()), parent.id());
    }

    public List<NodeRun> latest(NodeRun parent) {
        return latest(runs.nodes(parent.runId()), parent.id());
    }

    public static List<NodeRun> latest(List<NodeRun> all, String parentId) {
        Map<Integer, NodeRun> latest = new LinkedHashMap<>();
        all.stream().filter(item -> parentId.equals(item.parentNodeRunId()))
                .forEach(item -> latest.merge(item.itemIndex(), item,
                        (old, next) -> old.attempt() > next.attempt() ? old : next));
        return latest.values().stream().sorted(Comparator.comparingInt(NodeRun::itemIndex)).toList();
    }

    public Ports results(NodeRun parent, List<NodeRun> items) {
        String port = parent.plan().node().config().path("outputPort").asText();
        List<Value> output = new ArrayList<>();
        for (var item : items) {
            if (item.status() == State.SUCCEEDED || item.status() == State.PARTIAL_FAILED) {
                output.addAll(item.outputs().at(port));
            }
        }
        return new Ports(Map.of("results", List.copyOf(output)));
    }
}
