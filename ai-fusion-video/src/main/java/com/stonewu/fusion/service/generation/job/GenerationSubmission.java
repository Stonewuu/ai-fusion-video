package com.stonewu.fusion.service.generation.job;

import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;

import java.time.LocalDateTime;

/**
 * 内部执行资料，不直接作为控制器响应。
 */
public record GenerationSubmission(String id, String requestId, GenerationKind kind, long taskId,
                                   String publicTaskId, long userId, Long teamId, Long projectId,
                                   String canvasNodeRunId, ResourceCommandContext context, AiModel model,
                                   long apiConfigId, String apiConfigHash, String state,
                                   boolean cancelRequested, long revision, LocalDateTime createdAt) {
}
