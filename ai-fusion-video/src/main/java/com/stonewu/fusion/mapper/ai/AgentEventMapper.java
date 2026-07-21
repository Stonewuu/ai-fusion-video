package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentEvent;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface AgentEventMapper extends BaseMapper<AgentEvent> {

    @Select("""
            SELECT *
            FROM afv_agent_event FORCE INDEX (uk_agent_event_sequence)
            WHERE run_id = #{runId}
              AND sequence_no = #{sequenceNo}
            LIMIT 1
            """)
    AgentEvent selectByRunAndSequence(
            @Param("runId") String runId,
            @Param("sequenceNo") long sequenceNo);

    @Select("""
            SELECT *
            FROM afv_agent_event FORCE INDEX (idx_agent_event_publish)
            WHERE publish_required = 1
              AND publish_status = 'PENDING'
              AND (next_publish_attempt_at IS NULL
                   OR next_publish_attempt_at <= #{now})
            ORDER BY next_publish_attempt_at, id
            LIMIT #{limit}
            FOR UPDATE SKIP LOCKED
            """)
    List<AgentEvent> selectPendingPublishCandidatesForUpdate(
            @Param("now") LocalDateTime now,
            @Param("limit") int limit);

    @Select("""
            SELECT *
            FROM afv_agent_event FORCE INDEX (idx_agent_event_publish)
            WHERE publish_required = 1
              AND publish_status = 'CLAIMED'
              AND publish_claim_until <= #{now}
            ORDER BY id
            LIMIT #{limit}
            FOR UPDATE SKIP LOCKED
            """)
    List<AgentEvent> selectExpiredPublishCandidatesForUpdate(
            @Param("now") LocalDateTime now,
            @Param("limit") int limit);

    @Update("""
            UPDATE afv_agent_event
            SET publish_status = 'CLAIMED',
                publish_claim_owner = #{claimOwner},
                publish_claim_until = #{claimUntil},
                publish_attempts = publish_attempts + 1,
                next_publish_attempt_at = NULL
            WHERE id = #{eventId}
              AND publish_required = 1
              AND (
                    (publish_status = 'PENDING'
                     AND (next_publish_attempt_at IS NULL
                          OR next_publish_attempt_at <= #{now}))
                 OR (publish_status = 'CLAIMED'
                     AND publish_claim_until <= #{now})
              )
            """)
    int claimPublishCandidate(
            @Param("eventId") long eventId,
            @Param("claimOwner") String claimOwner,
            @Param("claimUntil") LocalDateTime claimUntil,
            @Param("now") LocalDateTime now);

    @Update("""
            UPDATE afv_agent_event
            SET publish_status = 'PUBLISHED',
                redis_published_at = UTC_TIMESTAMP(3),
                publish_claim_owner = NULL,
                publish_claim_until = NULL,
                next_publish_attempt_at = NULL,
                last_publish_error = NULL
            WHERE id = #{eventId}
              AND publish_status = 'CLAIMED'
              AND publish_claim_owner = #{claimOwner}
              AND publish_claim_until > UTC_TIMESTAMP(3)
            """)
    int markPublished(
            @Param("eventId") long eventId,
            @Param("claimOwner") String claimOwner);

    @Update("""
            UPDATE afv_agent_event
            SET publish_status = 'PENDING',
                publish_claim_owner = NULL,
                publish_claim_until = NULL,
                next_publish_attempt_at = GREATEST(
                        #{nextAttemptAt}, UTC_TIMESTAMP(3)),
                last_publish_error = #{lastPublishError}
            WHERE id = #{eventId}
              AND publish_status = 'CLAIMED'
              AND publish_claim_owner = #{claimOwner}
              AND publish_claim_until > UTC_TIMESTAMP(3)
            """)
    int releasePublishForRetry(
            @Param("eventId") long eventId,
            @Param("claimOwner") String claimOwner,
            @Param("nextAttemptAt") LocalDateTime nextAttemptAt,
            @Param("lastPublishError") String lastPublishError);
}
