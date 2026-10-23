package com.stonewu.fusion.service.canvas.run.mcp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.agentscope.mcp.McpWorkflowTool;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;

import java.time.Instant;

public final class CanvasMcpData {
    private CanvasMcpData() {
    }

    public enum Effect {READ_ONLY, WRITE, DESTRUCTIVE}

    public enum Idempotency {READ_ONLY, KEYED, NO_REPLAY}

    public enum OutputMode {TEXT, STRUCTURED_JSON, JSON_TEXT}

    public record Definition(McpWorkflowTool tool, Effect effect, Idempotency idempotency,
                             String idempotencyParameter, OutputMode outputMode, JsonNode outputSchema,
                             int timeoutSeconds) {
    }

    public record Contract(String id, long userId, String name, long version, boolean enabled,
                           Definition definition, String hash) {
    }

    public record Input(int schemaVersion, Contract contract, ObjectNode arguments, Ports sources) {
    }

    public record Claim(String id, String owner, long epoch) {
    }

    public record Job(String id, String nodeRunId, String runId, long userId, String contractId, Input input,
                      String status, Instant startedAt, ObjectNode result, boolean cancelled, boolean permitHeld,
                      String owner, long epoch, Instant leaseUntil, long revision, String error, Instant deadlineAt) {
        public Claim claim() {
            return new Claim(id, owner, epoch);
        }
    }
}
