package com.stonewu.fusion.controller.ai.vo;

import java.time.Instant;

public record AgentStateCleanupPolicyRespVO(
        int cleanupIntervalDays,
        int retentionDays,
        Instant nextCleanupAt,
        Instant lastCleanupAt) {
}
