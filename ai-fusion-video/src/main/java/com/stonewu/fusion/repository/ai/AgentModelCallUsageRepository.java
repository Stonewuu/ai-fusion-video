package com.stonewu.fusion.repository.ai;

import com.stonewu.fusion.entity.ai.AgentModelCallUsage;
import com.stonewu.fusion.enums.ai.AgentModelCallStatus;
import com.stonewu.fusion.service.ai.run.model.NormalizedModelUsage;

import java.time.Duration;
import java.time.Instant;
import java.util.List;

/** Synchronous transactional boundary for the durable model-call usage ledger. */
public interface AgentModelCallUsageRepository {

    void startCall(String runId, String modelCallId, String provider, String modelCode);

    boolean completeCall(
            String runId, String modelCallId, NormalizedModelUsage usage);

    boolean failCall(
            String runId, String modelCallId, AgentModelCallStatus status);

    /**
     * Claims ready rows. The owner is a per-attempt fencing token and must not be
     * reused for a later reclaim by the same process.
     */
    List<AgentModelCallUsage> claimSettlementBatch(
            String claimOwner, Duration claimLease, int limit);

    boolean markSettled(
            long usageId, String claimOwner, String downstreamSettlementId);

    boolean releaseSettlementForRetry(
            long usageId,
            String claimOwner,
            Instant nextAttemptAt,
            String sanitizedError);

    boolean markRunUsageSettledIfAllCallsSettled(String runId);
}
