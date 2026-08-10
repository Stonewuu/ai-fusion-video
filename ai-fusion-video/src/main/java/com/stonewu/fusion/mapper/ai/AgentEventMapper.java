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

    @Select("""
            SELECT COUNT(*)
            FROM afv_agent_event FORCE INDEX (idx_agent_event_publish)
            WHERE publish_required = 1
              AND publish_status <> 'PUBLISHED'
            """)
    long countOutstandingPublish();

    @Select("""
            SELECT *
            FROM afv_agent_event FORCE INDEX (uk_agent_event_sequence)
            WHERE run_id = #{runId}
              AND sequence_no > #{afterSequence}
              AND sequence_no <= #{throughSequence}
            ORDER BY sequence_no
            LIMIT #{limit}
            """)
    List<AgentEvent> selectProjectionRange(
            @Param("runId") String runId,
            @Param("afterSequence") long afterSequence,
            @Param("throughSequence") long throughSequence,
            @Param("limit") int limit);

    @Select("""
            SELECT *
            FROM afv_agent_event FORCE INDEX (uk_agent_event_sequence)
            WHERE run_id = #{runId}
              AND sequence_no > #{afterSequence}
              AND sequence_no <= #{throughSequence}
            ORDER BY sequence_no
            LIMIT #{limit}
            """)
    List<AgentEvent> selectReplayRange(
            @Param("runId") String runId,
            @Param("afterSequence") long afterSequence,
            @Param("throughSequence") long throughSequence,
            @Param("limit") int limit);

    @Select("""
            SELECT COALESCE(MAX(sequence_no), 0)
            FROM afv_agent_event FORCE INDEX (idx_agent_event_projection)
            WHERE run_id = #{runId}
              AND sequence_no < #{beforeSequence}
              AND output_type IS NOT NULL
              AND output_type NOT IN ('CONTENT', 'REASONING')
              AND (
                    parent_tool_call_id = #{parentToolCallId}
                 OR (parent_tool_call_id IS NULL AND #{parentToolCallId} IS NULL)
              )
            """)
    long selectLastContextBoundarySequence(
            @Param("runId") String runId,
            @Param("beforeSequence") long beforeSequence,
            @Param("parentToolCallId") String parentToolCallId);

    @Select("""
            SELECT *
            FROM afv_agent_event FORCE INDEX (uk_agent_event_sequence)
            WHERE run_id = #{runId}
              AND sequence_no > #{afterSequence}
              AND sequence_no < #{beforeSequence}
              AND output_type IN ('CONTENT', 'REASONING')
              AND (
                    parent_tool_call_id = #{parentToolCallId}
                 OR (parent_tool_call_id IS NULL AND #{parentToolCallId} IS NULL)
              )
            ORDER BY sequence_no
            """)
    List<AgentEvent> selectContextDeltas(
            @Param("runId") String runId,
            @Param("afterSequence") long afterSequence,
            @Param("beforeSequence") long beforeSequence,
            @Param("parentToolCallId") String parentToolCallId);

    @Select("""
            SELECT *
            FROM afv_agent_event FORCE INDEX (uk_agent_event_sequence)
            WHERE run_id = #{runId}
              AND tool_call_id = #{toolCallId}
              AND sequence_no <= #{throughSequence}
              AND raw_event_type IN (
                    'TOOL_CALL_DELTA',
                    'TOOL_RESULT_TEXT_DELTA',
                    'TOOL_RESULT_DATA_DELTA')
            ORDER BY sequence_no
            """)
    List<AgentEvent> selectToolDeltas(
            @Param("runId") String runId,
            @Param("toolCallId") String toolCallId,
            @Param("throughSequence") long throughSequence);

    @Select("""
            SELECT *
            FROM afv_agent_event FORCE INDEX (uk_agent_event_sequence)
            WHERE run_id = #{runId}
              AND reply_id = #{replyId}
              AND raw_event_type = 'REQUIRE_USER_CONFIRM'
              AND source = 'platform/waiting-candidate'
            ORDER BY sequence_no DESC
            LIMIT 1
            """)
    AgentEvent selectConfirmationCandidate(
            @Param("runId") String runId,
            @Param("replyId") String replyId);

    @Select("""
            SELECT *
            FROM afv_agent_event FORCE INDEX (uk_agent_event_sequence)
            WHERE run_id = #{runId}
              AND raw_event_type = 'REQUIRE_USER_CONFIRM'
              AND source = 'platform/waiting-candidate'
            ORDER BY sequence_no DESC
            LIMIT 1
            """)
    AgentEvent selectLatestConfirmationCandidate(@Param("runId") String runId);

    @Select("""
            SELECT *
            FROM afv_agent_event FORCE INDEX (uk_agent_event_sequence)
            WHERE run_id = #{runId}
              AND tool_call_id = #{toolCallId}
              AND raw_event_type = 'PLATFORM_REQUIRE_EXTERNAL_EXECUTION'
              AND source = 'platform/waiting'
            ORDER BY sequence_no DESC
            LIMIT 1
            """)
    AgentEvent selectPendingExternalExecution(
            @Param("runId") String runId,
            @Param("toolCallId") String toolCallId);
}
