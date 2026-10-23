package com.stonewu.fusion.service.ai.comfyui.test;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiJobResult;
import com.stonewu.fusion.service.media.job.MediaJob;

import java.time.LocalDateTime;

/**
 * 内部记录包含执行快照，禁止作为 HTTP 响应直接序列化。
 */
public record ComfyUiTestRun(MediaJob job, long workflowId, long versionId, String workflowHash,
                             long apiConfigId, String apiConfigHash, String promptId,
                             ObjectNode request, LocalDateTime submittedAt, LocalDateTime acknowledgedAt,
                             ComfyUiJobResult result, boolean permitHeld, boolean cancelSent,
                             int pollAttempts, int consecutiveErrors) {
}
