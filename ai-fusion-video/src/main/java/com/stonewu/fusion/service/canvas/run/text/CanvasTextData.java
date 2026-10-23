package com.stonewu.fusion.service.canvas.run.text;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.canvas.run.CanvasRunData.Ports;

import java.time.Instant;

public final class CanvasTextData {
    private CanvasTextData() {
    }

    public record Input(int schemaVersion, String nodeRunId, AiModel model, String providerHash,
                        String instruction, Ports inputs) {
    }

    public record Claim(String id, String owner, long epoch) {
    }

    public record Job(String id, String nodeRunId, String runId, Long projectId, Input input, String status,
                      String phase, boolean cancelRequested, boolean permitHeld, Instant startedAt,
                      ObjectNode result, boolean resultAccepted, long revision, String error, String owner, long epoch,
                      Instant leaseUntil, Instant deadlineAt) {
        public Claim claim() {
            return new Claim(id, owner, epoch);
        }
    }
}
