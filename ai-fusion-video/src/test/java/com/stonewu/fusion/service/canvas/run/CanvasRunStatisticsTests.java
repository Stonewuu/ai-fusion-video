package com.stonewu.fusion.service.canvas.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CanvasRunStatisticsTests {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Instant START = Instant.parse("2026-10-14T00:00:00Z");

    @Test void distinguishesGraphResultsRetriesAndPublishedMediaWithoutCountingInputsOrReuseAgain() {
        Value video = new Value("output", CanvasValueType.VIDEO, null, "video", 1, null,
                JSON.createObjectNode().set("content", JSON.valueToTree(Map.of("byteSize", 1024, "durationMs", 2500))), null);
        var output = new Ports(Map.of("media", List.of(video, video)));
        var nodes = List.of(
                node(State.SUCCEEDED, null, 1, "EXECUTE", "MEDIA", output),
                node(State.SUCCEEDED, null, 1, "EXECUTE", null, output),
                node(State.SUCCEEDED, null, 1, "REUSE", "MEDIA", output),
                node(State.BLOCKED, null, 1, "EXECUTE", null, Ports.empty()),
                node(State.CANCELLED, null, 1, "EXECUTE", null, Ports.empty()),
                node(State.PARTIAL_FAILED, null, 1, "EXECUTE", "BATCH", output),
                node(State.FAILED, "batch", 1, "EXECUTE", "MEDIA", Ports.empty()),
                node(State.SUCCEEDED, "batch", 2, "EXECUTE", "MEDIA", output));
        var measured = CanvasRunStatistics.measure(run(State.PARTIAL_FAILED), nodes, List.of(), START.plusSeconds(100), Set.of());
        assertThat(measured.nodes()).isEqualTo(new CanvasRunStatistics.Nodes(6, 3, 1, 1, 1, 0, 1, 2, 1));
        assertThat(measured.media()).isEqualTo(new CanvasRunStatistics.Media(0, 1, 0, 2500, 0, 0, 1024, 0));
        assertThat(measured.elapsedMs()).isEqualTo(5000);
        assertThat(measured.tokens().total()).isNull();
    }

    @Test void unknownAndMalformedUsageAreExplicitlyIncompleteWhileZeroIsMeasured() {
        List<JsonNode> usage = List.of(
                JSON.valueToTree(Map.of("inputTokens", 10, "outputTokens", 5, "totalTokens", 15)),
                JSON.valueToTree(Map.of("inputTokens", 0, "outputTokens", 0, "totalTokens", 0)),
                JSON.valueToTree(Map.of()),
                JSON.valueToTree(Map.of("inputTokens", -1, "outputTokens", 5, "totalTokens", 4)),
                JSON.valueToTree(Map.of("inputTokens", "10", "outputTokens", 5, "totalTokens", 15)));
        var measured = CanvasRunStatistics.measure(run(State.RUNNING),
                List.of(node(State.WAITING_SELECTION, null, 1, "EXECUTE", null, Ports.empty())), usage, START.plusSeconds(20), Set.of());
        assertThat(measured.tokens()).isEqualTo(new CanvasRunStatistics.Tokens(5, 2, 10L, 5L, 15L));
        assertThat(measured.elapsedMs()).isEqualTo(20000);
        assertThat(measured.nodes().pending()).isEqualTo(1);
    }

    private Run run(State status) {
        var run = mock(Run.class);
        when(run.status()).thenReturn(status);
        when(run.createdAt()).thenReturn(START);
        when(run.updatedAt()).thenReturn(START.plusSeconds(5));
        return run;
    }

    @Test void crossRunTotalsDeduplicateMediaAndKeepUnknownUsageDistinctFromZero() {
        var accumulator = new CanvasActivityStatistics.Accumulator();
        Value video = new Value("output", CanvasValueType.VIDEO, null, "same-video", 1, null,
                JSON.createObjectNode().set("content", JSON.valueToTree(Map.of("byteSize", 1024, "durationMs", 2500))), null);
        var outputs = new Ports(Map.of("video", List.of(video)));
        var produced = node(State.SUCCEEDED,null,1,"EXECUTE","MEDIA",outputs);
        var reused = node(State.SUCCEEDED,null,1,"REUSE","MEDIA",outputs);
        var unavailable = node(State.SUCCEEDED,null,1,"EXECUTE","MEDIA",new Ports(Map.of("video",List.of(
                new Value("other",CanvasValueType.VIDEO,null,"private",1,null,video.data(),null)))));
        var first = CanvasRunStatistics.measure(run(State.SUCCEEDED),List.of(produced),List.of(),START.plusSeconds(10),Set.of());
        var second = CanvasRunStatistics.measure(run(State.SUCCEEDED),List.of(reused),
                List.of(JSON.valueToTree(Map.of("inputTokens",0,"outputTokens",0,"totalTokens",0)), JSON.createObjectNode()),START.plusSeconds(10),Set.of());
        accumulator.add("SUCCEEDED",first); accumulator.add("SUCCEEDED",second);
        CanvasRunStatistics.collectProduced(produced,Set.of(),accumulator.produced);
        CanvasRunStatistics.collectProduced(produced,Set.of(),accumulator.produced);
        CanvasRunStatistics.collectProduced(reused,Set.of(),accumulator.produced);
        CanvasRunStatistics.collectProduced(unavailable,Set.of("node"),accumulator.produced);
        assertThat(accumulator.nodes.total()).isEqualTo(2);
        assertThat(accumulator.nodes.reused()).isEqualTo(1);
        assertThat(accumulator.elapsed).isEqualTo(10000);
        assertThat(accumulator.tokens()).isEqualTo(new CanvasRunStatistics.Tokens(2,1,0L,0L,0L));
        assertThat(CanvasRunStatistics.media(accumulator.produced,1)).isEqualTo(new CanvasRunStatistics.Media(0,1,0,2500,0,0,1024,1));
    }
    private NodeRun node(State status, String parent, int attempt, String action, String external, Ports outputs) {
        var node = mock(NodeRun.class);
        when(node.id()).thenReturn("node");
        var plan = mock(NodePlan.class);
        when(plan.action()).thenReturn(action);
        when(node.plan()).thenReturn(plan);
        when(node.status()).thenReturn(status);
        when(node.parentNodeRunId()).thenReturn(parent);
        when(node.attempt()).thenReturn(attempt);
        when(node.externalId()).thenReturn(external == null ? null : "task");
        when(node.externalKind()).thenReturn(external);
        when(node.outputs()).thenReturn(outputs);
        return node;
    }
}
