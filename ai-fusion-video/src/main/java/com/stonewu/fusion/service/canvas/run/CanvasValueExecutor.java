package com.stonewu.fusion.service.canvas.run;

import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;
import static com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Executor;
import static com.stonewu.fusion.service.canvas.run.CanvasRunExecutors.Outcome;

@Component
@RequiredArgsConstructor
public class CanvasValueExecutor implements Executor {
    private final CanvasRunSources sources;

    @Override
    public Set<CanvasNodeKind> kinds() {
        return Set.of(CanvasNodeKind.TEXT, CanvasNodeKind.STORY, CanvasNodeKind.TEXT_COMBINE,
                CanvasNodeKind.COLLECTION, CanvasNodeKind.SELECTION, CanvasNodeKind.IMAGE, CanvasNodeKind.VIDEO,
                CanvasNodeKind.AUDIO, CanvasNodeKind.DOCUMENT, CanvasNodeKind.TEXT_FILE, CanvasNodeKind.PROJECT,
                CanvasNodeKind.SCRIPT, CanvasNodeKind.SCRIPT_EPISODE, CanvasNodeKind.SCRIPT_SCENE,
                CanvasNodeKind.STORYBOARD, CanvasNodeKind.STORYBOARD_EPISODE, CanvasNodeKind.STORYBOARD_SCENE,
                CanvasNodeKind.SHOT, CanvasNodeKind.ASSET, CanvasNodeKind.ASSET_ITEM);
    }

    @Override
    public Outcome step(Run run, NodeRun node, Claim claim) {
        var kind = node.plan().node().kind();
        if (kind == CanvasNodeKind.TEXT_COMBINE) {
            String separator = node.plan().node().config().path("separator").asText("\n");
            String result = node.inputs().at("texts").stream().map(Value::text).collect(Collectors.joining(separator));
            if (result.length() > 128000) {
                throw CanvasRunException.invalid("拼接后的文本超过 128000 字符，请缩小输入范围");
            }
            return Outcome.done(new Ports(Map.of("text", List.of(sources.text(result)))));
        }
        if (kind == CanvasNodeKind.COLLECTION) {
            return Outcome.done(new Ports(Map.of("items", node.inputs().at("items"))));
        }
        if (kind == CanvasNodeKind.SELECTION) {
            String id = node.plan().node().config().path("selectedOutputId").asText("");
            if (id.isBlank()) {
                return new Outcome(State.WAITING_SELECTION, Ports.empty(), null, null, null);
            }
            var selected = node.inputs().at("candidates").stream().filter(value -> value.id().equals(id)).findFirst()
                    .orElseThrow(() -> CanvasRunException.invalid("预选结果不在本次冻结的候选中，请清除预选后重新运行"));
            return Outcome.done(new Ports(Map.of("selected", List.of(selected))));
        }
        return Outcome.done(node.plan().sourceOutputs());
    }
}
