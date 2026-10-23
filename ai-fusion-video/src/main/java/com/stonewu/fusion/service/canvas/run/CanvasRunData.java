package com.stonewu.fusion.service.canvas.run;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.*;
import com.stonewu.fusion.service.generation.job.GenerationClientScope;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 持久化计划只由服务端生成；HTTP 不能提供工作者身份、输出或私有模型快照。
 */
public final class CanvasRunData {
    private CanvasRunData() {
    }

    public enum Scope {SINGLE, SELECTION, DEPENDENCIES, ALL, STALE}

    public enum State {
        QUEUED, WAITING_INPUT, READY, RUNNING, WAITING_SELECTION, WAITING_CONFIRMATION,
        UNKNOWN_SUBMISSION, NEEDS_REVIEW, CANCEL_REQUESTED, SUCCEEDED, PARTIAL_FAILED,
        FAILED, BLOCKED, CANCELLED;

        public boolean terminal() {
            return Set.of(SUCCEEDED, PARTIAL_FAILED, FAILED, BLOCKED, CANCELLED).contains(this);
        }
    }

    public record Budget(int maxNodes, int maxImages, int maxVideoSeconds, int maxParallel, int maxBatchItems) {
    }

    public record PlanRequest(@NotNull Long expectedRevision, @NotNull Scope mode,
                              @NotNull List<String> nodeIds, @NotNull Map<String, String> reuse,
                              @NotNull @Valid Budget budget, @NotNull @Valid GenerationClientScope scope) {
    }

    public record Start(@NotBlank @Pattern(regexp = "[A-Za-z0-9:._-]{1,96}") String requestId,
                        @NotBlank String planId, @NotBlank String planHash,
                        @NotNull @Valid GenerationClientScope scope) {
    }

    public record Selection(@NotNull Long expectedRevision, @NotBlank String outputId,
                            @NotNull @Valid GenerationClientScope scope) {
    }

    public record Issue(String nodeId, String code, String message) {
    }

    public record Cost(int nodes, int images, int videoSeconds, int batchItems, int agentRuns, int mcpCalls,
                       String amount) {
        public Cost(int nodes, int images, int videoSeconds, int batchItems, int agentRuns, String amount) {
            this(nodes, images, videoSeconds, batchItems, agentRuns, 0, amount);
        }

        public Cost(int nodes, int images, int videoSeconds, int batchItems, String amount) {
            this(nodes, images, videoSeconds, batchItems, 0, 0, amount);
        }
    }

    public record Value(String id, CanvasValueType type, String text, String mediaId, Integer mediaVersion,
                        CanvasResourceReference resource, ObjectNode data, String sourceNodeRunId) {
    }

    public record Ports(Map<String, List<Value>> values) {
        public Ports {
            values = Map.copyOf(values);
        }

        public static Ports empty() {
            return new Ports(Map.of());
        }

        public List<Value> at(String port) {
            return values.getOrDefault(port, List.of());
        }

        public List<Value> all() {
            return values.values().stream().flatMap(List::stream).toList();
        }
    }

    public record NodePlan(CanvasNode node, String fingerprint, String action, String reusedNodeRunId,
                           List<String> dependencies, List<CanvasEdge> incoming, Map<String, Value> pinned,
                           Ports sourceOutputs, ObjectNode environment, int imageCount, int videoSeconds) {
    }

    public record Plan(String id, String canvasId, long documentRevision, Scope mode, List<String> requested,
                       CanvasActor actor, CanvasGraph graph, List<NodePlan> nodes, Budget budget,
                       Cost cost, List<Issue> issues, Instant expiresAt) {
    }

    public record ExternalOperation(String name, String serverName, String toolName, String effect,
                                    String idempotency, int timeoutSeconds, int maximumCalls) {
    }

    public record PlanNode(String nodeId, String title, String kind, String action, String reusedNodeRunId,
                           List<String> dependencies, int imageCount, int videoSeconds,
                           ExternalOperation externalOperation) {
    }

    public record PlanView(String id, String canvasId, long documentRevision, String hash, Scope mode,
                           List<PlanNode> nodes, Budget budget, Cost cost, List<Issue> issues, Instant expiresAt) {
    }

    public record Run(String id, String canvasId, CanvasActor actor, ResourceCommandContext context,
                      Plan plan, State status, boolean cancelRequested, long revision, String error,
                      Instant deadlineAt, Instant createdAt, Instant updatedAt) {
    }

    public record NodeRun(String id, String runId, String nodeId, String parentNodeRunId, int itemIndex, int attempt,
                          NodePlan plan, Ports inputs, Ports outputs,
                          ObjectNode checkpoint, State status, String error, String externalKind, String externalId,
                          long revision, String owner, long epoch, Instant leaseUntil, Instant updatedAt) {
    }

    public record Claim(String id, String runId, String owner, long epoch) {
    }

    public record NodeView(String id, String nodeId, String title, String kind, String action, State status,
                           String error, long revision, Ports inputs, Ports outputs,
                           String externalKind, String externalId, String derivedState,
                           String parentNodeRunId, int itemIndex, int attempt, ObjectNode writeback,
                           boolean contentUnavailable) {
    }

    public record View(String id, String canvasId, long documentRevision, Scope mode, State status,
                       boolean cancelRequested, long revision, String error, Cost cost,
                       Instant createdAt, Instant updatedAt, List<NodeView> nodes, boolean canControl,
                       CanvasRunStatistics.Statistics statistics) {
    }

    public record Summary(String id, State status, Scope mode, long revision, Instant createdAt, int nodes) {
    }

    public record NodeActivity(String runId, State runStatus, boolean canControl, NodeView node,
                               List<NodeView> children) {
    }
}
