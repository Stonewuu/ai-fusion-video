package com.stonewu.fusion.controller.ai.vo;

import java.time.Instant;

/** Refresh-safe snapshot of an active durable root run. */
public record RunningPipelineRunRespVO(
        String runId,
        String conversationId,
        Long projectId,
        String title,
        String category,
        String status,
        long lastSequence,
        String waitingReplyId,
        Instant startedAt) {
}
