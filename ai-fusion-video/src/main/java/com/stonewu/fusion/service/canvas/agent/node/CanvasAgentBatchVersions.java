package com.stonewu.fusion.service.canvas.agent.node;

import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasResourceReference;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.NodeRun;
import com.stonewu.fusion.service.canvas.run.CanvasRunException;
import lombok.RequiredArgsConstructor;
import com.stonewu.fusion.mapper.canvas.CanvasAgentNodeJobMapper;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 专业批量逐项运行，只接续本批较早子项真实提交的版本，不接纳用户或其他运行的新版本。
 */
@Component
@RequiredArgsConstructor
public class CanvasAgentBatchVersions {
    private final CanvasAgentNodeJobMapper jobs;
    private final CanvasJson json;

    public Map<String, CanvasAgentNodeScope.Entry> previous(NodeRun node) {
        Map<String, CanvasAgentNodeScope.Entry> result = new LinkedHashMap<>();
        if (node.parentNodeRunId() == null) {
            return result;
        }
        var snapshots = jobs.previousScopes(node.parentNodeRunId(), node.attempt(), node.itemIndex());
        for (String snapshot : snapshots) {
            json.read(snapshot, CanvasAgentNodeScope.class).resources().forEach((key, entry) -> {
                if (entry.changed()) {
                    result.merge(key, entry, (old, next) -> old.version() > next.version() ? old : next);
                }
            });
        }
        return result;
    }

    public CanvasResourceReference advance(CanvasResourceReference original,
                                           Map<String, CanvasAgentNodeScope.Entry> previous) {
        var entry = previous.get(CanvasAgentNodeScope.key(original.type(), original.id()));
        if (entry == null) {
            return original;
        }
        if (entry.deleted()) {
            throw CanvasRunException.conflict("本批前一项已删除后续项需要的业务对象，请重新选择范围");
        }
        return new CanvasResourceReference(original.type(), original.id(), (long) entry.version());
    }
}
