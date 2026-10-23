package com.stonewu.fusion.service.ai.comfyui.test;

import com.stonewu.fusion.service.ai.comfyui.ComfyUiStoredOutput;

import java.time.Instant;
import java.util.List;
import java.util.Map;

public record ComfyUiTestView(String id, long versionId, String workflowHash, String status,
                              String phase, String error, String promptId, boolean acknowledged,
                              boolean cancelRequested, boolean permitHeld, long revision,
                              Instant createdAt, Instant deadlineAt, List<ComfyUiStoredOutput> outputs,
                              Map<String, String> inputSummary, List<String> availableActions) {
}
