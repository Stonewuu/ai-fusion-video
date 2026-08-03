package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentModelCallUsage;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface AgentModelCallUsageMapper extends BaseMapper<AgentModelCallUsage> {

    @Select("""
            SELECT *
            FROM afv_agent_model_call_usage FORCE INDEX (uk_agent_usage_call)
            WHERE run_id = #{runId}
              AND model_call_id = #{modelCallId}
            LIMIT 1
            """)
    AgentModelCallUsage selectByRunAndCall(
            @Param("runId") String runId,
            @Param("modelCallId") String modelCallId);

    @Update("""
            UPDATE afv_agent_model_call_usage
            SET status = 'COMPLETED',
                input_tokens = #{inputTokens},
                output_tokens = #{outputTokens},
                reasoning_tokens = #{reasoningTokens},
                cache_tokens = #{cacheTokens},
                usage_json = #{usageJson},
                finished_at = #{finishedAt},
                update_time = #{finishedAt}
            WHERE run_id = #{runId}
              AND model_call_id = #{modelCallId}
              AND status = 'STARTED'
            """)
    int completeStarted(
            @Param("runId") String runId,
            @Param("modelCallId") String modelCallId,
            @Param("inputTokens") Long inputTokens,
            @Param("outputTokens") Long outputTokens,
            @Param("reasoningTokens") Long reasoningTokens,
            @Param("cacheTokens") Long cacheTokens,
            @Param("usageJson") String usageJson,
            @Param("finishedAt") LocalDateTime finishedAt);

    @Update("""
            UPDATE afv_agent_model_call_usage
            SET status = #{status},
                finished_at = #{finishedAt},
                update_time = #{finishedAt}
            WHERE run_id = #{runId}
              AND model_call_id = #{modelCallId}
              AND status = 'STARTED'
            """)
    int finishStarted(
            @Param("runId") String runId,
            @Param("modelCallId") String modelCallId,
            @Param("status") String status,
            @Param("finishedAt") LocalDateTime finishedAt);

    @Update("""
            UPDATE afv_agent_model_call_usage
            SET status = #{status},
                finished_at = #{finishedAt},
                update_time = #{finishedAt}
            WHERE run_id = #{runId}
              AND status = 'STARTED'
            """)
    int finishAllStartedForRun(
            @Param("runId") String runId,
            @Param("status") String status,
            @Param("finishedAt") LocalDateTime finishedAt);

    @Select("""
            SELECT *
            FROM afv_agent_model_call_usage FORCE INDEX (idx_agent_usage_settlement)
            WHERE status IN ('COMPLETED', 'FAILED', 'CANCELLED')
              AND (
                    (settlement_status = 'PENDING'
                     AND (next_settlement_attempt_at IS NULL
                          OR next_settlement_attempt_at <= #{now}))
                 OR (settlement_status = 'CLAIMED'
                     AND settlement_claim_until <= #{now})
              )
            ORDER BY id
            LIMIT #{limit}
            FOR UPDATE
            """)
    List<AgentModelCallUsage> selectSettlementCandidatesForUpdate(
            @Param("now") LocalDateTime now,
            @Param("limit") int limit);

    @Update("""
            UPDATE afv_agent_model_call_usage
            SET settlement_status = 'CLAIMED',
                settlement_claim_owner = #{claimOwner},
                settlement_claim_until = #{claimUntil},
                settlement_attempts = settlement_attempts + 1,
                next_settlement_attempt_at = NULL,
                update_time = #{now}
            WHERE id = #{usageId}
              AND status IN ('COMPLETED', 'FAILED', 'CANCELLED')
              AND (
                    (settlement_status = 'PENDING'
                     AND (next_settlement_attempt_at IS NULL
                          OR next_settlement_attempt_at <= #{now}))
                 OR (settlement_status = 'CLAIMED'
                     AND settlement_claim_until <= #{now})
              )
            """)
    int claimCandidate(
            @Param("usageId") long usageId,
            @Param("claimOwner") String claimOwner,
            @Param("claimUntil") LocalDateTime claimUntil,
            @Param("now") LocalDateTime now);

    @Update("""
            UPDATE afv_agent_model_call_usage
            SET settlement_status = 'SETTLED',
                settlement_claim_owner = NULL,
                settlement_claim_until = NULL,
                next_settlement_attempt_at = NULL,
                downstream_settlement_id = #{downstreamSettlementId},
                last_settlement_error = NULL,
                update_time = UTC_TIMESTAMP(3)
            WHERE id = #{usageId}
              AND settlement_status = 'CLAIMED'
              AND settlement_claim_owner = #{claimOwner}
              AND settlement_claim_until > UTC_TIMESTAMP(3)
            """)
    int markSettled(
            @Param("usageId") long usageId,
            @Param("claimOwner") String claimOwner,
            @Param("downstreamSettlementId") String downstreamSettlementId);

    @Update("""
            UPDATE afv_agent_model_call_usage
            SET settlement_status = 'PENDING',
                settlement_claim_owner = NULL,
                settlement_claim_until = NULL,
                next_settlement_attempt_at = GREATEST(
                        #{nextAttemptAt}, UTC_TIMESTAMP(3)),
                last_settlement_error = #{lastSettlementError},
                update_time = UTC_TIMESTAMP(3)
            WHERE id = #{usageId}
              AND settlement_status = 'CLAIMED'
              AND settlement_claim_owner = #{claimOwner}
              AND settlement_claim_until > UTC_TIMESTAMP(3)
            """)
    int releaseSettlementForRetry(
            @Param("usageId") long usageId,
            @Param("claimOwner") String claimOwner,
            @Param("nextAttemptAt") LocalDateTime nextAttemptAt,
            @Param("lastSettlementError") String lastSettlementError);

    @Update("""
            UPDATE afv_agent_run r
            SET r.usage_settled = 1,
                r.usage_settled_at = #{settledAt},
                r.update_time = #{settledAt}
            WHERE r.run_id = #{runId}
              AND r.usage_settled = 0
              AND r.status IN ('COMPLETED', 'FAILED', 'CANCELLED')
              AND NOT EXISTS (
                    SELECT 1
                    FROM afv_agent_model_call_usage u
                    WHERE u.run_id = r.run_id
                      AND u.settlement_status <> 'SETTLED'
              )
            """)
    int markRunUsageSettledIfNoUnsettledCall(
            @Param("runId") String runId,
            @Param("settledAt") LocalDateTime settledAt);
}
