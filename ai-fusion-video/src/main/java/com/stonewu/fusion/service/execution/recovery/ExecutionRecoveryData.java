package com.stonewu.fusion.service.execution.recovery;

import com.stonewu.fusion.service.canvas.model.CanvasActor;
import jakarta.validation.constraints.*;

import java.time.Instant;
import java.util.List;

public final class ExecutionRecoveryData {
    private ExecutionRecoveryData() {
    }

    public enum Kind {GENERATION, TEXT, CANVAS_MEDIA, COMFY_TEST, EPISODE_AUDIO, AGENT_MODEL, MCP}

    public record Ref(Kind kind, String id) {
    }

    public record Remote(String step, String attemptId, String providerTaskId, boolean acknowledged,
                         boolean responseRecorded) {
    }

    public record Target(Kind kind, String id, String status, long revision, CanvasActor actor, Long projectId,
                         String canvasId, String runId, boolean capacityHeld, boolean cancelled, List<Remote> remote) {
    }

}
