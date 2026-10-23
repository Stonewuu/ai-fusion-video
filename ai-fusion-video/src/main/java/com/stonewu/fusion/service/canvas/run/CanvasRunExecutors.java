package com.stonewu.fusion.service.canvas.run;

import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import org.springframework.stereotype.Component;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;

@Component
public class CanvasRunExecutors {
    public interface Executor {
        Set<CanvasNodeKind> kinds();

        Outcome step(Run run, NodeRun node, Claim claim) throws Exception;

        default void cancel(Run run, NodeRun node) {
        }
    }

    public record Outcome(State state, Ports outputs, String externalKind, String externalId, String error) {
        public static Outcome done(Ports outputs) {
            return new Outcome(State.SUCCEEDED, outputs, null, null, null);
        }
    }

    private final Map<CanvasNodeKind, Executor> entries;

    public CanvasRunExecutors(List<Executor> executors) {
        Map<CanvasNodeKind, Executor> result = new EnumMap<>(CanvasNodeKind.class);
        for (var executor : executors) {
            for (var kind : executor.kinds()) {
                if (result.putIfAbsent(kind, executor) != null) {
                    throw new IllegalStateException("节点执行器重复登记：" + kind);
                }
            }
        }
        entries = Map.copyOf(result);
    }

    public boolean supports(CanvasNodeKind kind) {
        return entries.containsKey(kind);
    }

    public Executor require(CanvasNodeKind kind) {
        var executor = entries.get(kind);
        if (executor == null) {
            throw CanvasRunException.invalid("节点执行器尚未接通：" + kind.code());
        }
        return executor;
    }
}
