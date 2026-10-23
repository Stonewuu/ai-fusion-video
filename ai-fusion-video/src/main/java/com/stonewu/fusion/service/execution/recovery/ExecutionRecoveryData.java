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

    public record View(Kind kind, String id, String status, long revision, long actorUserId, Long actorTeamId,
                       Long projectId, String canvasId, boolean capacityHeld, boolean cancelRequested,
                       List<Remote> remote, String factsHash, boolean canResolve, String reason) {
    }

    public record Decision(@NotNull @Positive Long adminUserId,
                           @NotBlank @Pattern(regexp = "[A-Za-z0-9._:-]{1,128}") String requestId,
                           @NotNull @PositiveOrZero Long expectedRevision,
                           @NotBlank @Pattern(regexp = "[a-f0-9]{64}") String expectedFactsHash,
                           @NotBlank @Size(max = 2000) String note,
                           @AssertTrue boolean providerFinished) {
    }

    public record Receipt(String decisionId, Kind kind, String jobId, String status, long revision) {
    }

    public record History(String decisionId, long adminUserId, Kind kind, String jobId,
                          long actorUserId, String note, Instant createdAt) {
    }

    public record Page(List<View> items, boolean hasMore, int page) {
    }
}
