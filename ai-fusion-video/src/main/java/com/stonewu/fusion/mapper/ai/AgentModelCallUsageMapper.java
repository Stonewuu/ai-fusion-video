package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentModelCallUsage;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface AgentModelCallUsageMapper extends BaseMapper<AgentModelCallUsage> {

    AgentModelCallUsage selectByRunAndCall(
            @Param("runId") String runId,
            @Param("modelCallId") String modelCallId);

    List<AgentModelCallUsage> selectSettlementCandidatesForUpdate(
            @Param("now") LocalDateTime now,
            @Param("limit") int limit);

    int claimCandidate(
            @Param("usageId") long usageId,
            @Param("claimOwner") String claimOwner,
            @Param("claimUntil") LocalDateTime claimUntil,
            @Param("now") LocalDateTime now);

    int markSettled(
            @Param("usageId") long usageId,
            @Param("claimOwner") String claimOwner,
            @Param("downstreamSettlementId") String downstreamSettlementId);

    int releaseSettlementForRetry(
            @Param("usageId") long usageId,
            @Param("claimOwner") String claimOwner,
            @Param("nextAttemptAt") LocalDateTime nextAttemptAt,
            @Param("lastSettlementError") String lastSettlementError);

    int markRunUsageSettledIfNoUnsettledCall(
            @Param("runId") String runId,
            @Param("settledAt") LocalDateTime settledAt);

    int completeObserved(@Param("input") Integer input, @Param("output") Integer output, @Param("cache") Integer cache,
                         @Param("usage") String usage, @Param("runId") String runId, @Param("callId") String callId);

    int adminCancel(@Param("id") String id);
}
