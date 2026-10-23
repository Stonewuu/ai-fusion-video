package com.stonewu.fusion.service.generation.job;

import com.fasterxml.jackson.databind.node.ObjectNode;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

public record RemoteGenerationJob(String id, String submissionId, long itemId, int itemIndex,
                                  long modelId, long userId, Long projectId, String state, String phase,
                                  String providerTaskId, String activeAttemptId, ObjectNode input,
                                  RemoteGenerationResult result, boolean cancelRequested, boolean cancelSent,
                                  boolean permitHeld, String owner, long epoch, LocalDateTime leaseUntil,
                                  LocalDateTime deadline, int pollAttempts, int consecutiveErrors,
                                  String errorCode, String errorMessage, long revision) {
    public record Claim(String id, String owner, long epoch) {
    }

    public Claim claim() {
        return new Claim(id, owner, epoch);
    }

    public boolean terminal() {
        return state.equals("SUCCEEDED") || state.equals("PARTIAL_FAILED") || state.equals("FAILED") || state.equals(
                "CANCELLED");
    }

    public boolean requiresReview() {
        return state.equals("UNKNOWN_SUBMISSION") || state.equals("NEEDS_REVIEW");
    }

    public List<Long> itemIds() {
        var ids = input.get("_resultItemIds");
        if (ids == null || !ids.isArray() || ids.isEmpty() || ids.size() > 32) {
            throw new IllegalStateException("生成执行缺少固定结果条目集合");
        }
        var result = new ArrayList<Long>();
        for (var id : ids) {
            if (!id.isIntegralNumber() || id.asLong() < 1 || result.contains(id.asLong())) {
                throw new IllegalStateException("生成结果条目集合无效");
            }
            result.add(id.asLong());
        }
        return List.copyOf(result);
    }
}
