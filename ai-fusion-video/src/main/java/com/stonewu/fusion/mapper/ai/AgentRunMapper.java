package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentRun;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface AgentRunMapper extends BaseMapper<AgentRun> {

    @Select("""
            SELECT *
            FROM afv_agent_run FORCE INDEX (uk_agent_run_id)
            WHERE run_id = #{runId}
            LIMIT 1
            FOR UPDATE
            """)
    AgentRun selectByRunIdForUpdate(@Param("runId") String runId);

    @Select("""
            SELECT *
            FROM afv_agent_run FORCE INDEX (uk_agent_run_id)
            WHERE run_id = #{runId}
            LIMIT 1
            """)
    AgentRun selectByRunId(@Param("runId") String runId);

    @Select("""
            SELECT *
            FROM afv_agent_run
            WHERE terminal_sequence IS NOT NULL
              AND (
                projected_through_sequence < terminal_sequence
                OR projection_completed_at IS NULL
              )
            ORDER BY id
            LIMIT #{limit}
            """)
    List<AgentRun> selectProjectionRecoveryCandidates(@Param("limit") int limit);

    @Select("""
            SELECT r.*
            FROM afv_agent_run r FORCE INDEX (uk_agent_run_id)
            INNER JOIN afv_agent_conversation c
              ON c.conversation_id = r.conversation_id
             AND c.deleted = 0
            WHERE r.run_id = #{runId}
              AND r.user_id = #{userId}
              AND c.user_id = #{userId}
            LIMIT 1
            """)
    AgentRun selectAuthorizedByRunId(
            @Param("runId") String runId,
            @Param("userId") long userId);

    @Select("""
            SELECT r.*
            FROM afv_agent_run r FORCE INDEX (idx_agent_run_conversation_status)
            INNER JOIN afv_agent_conversation c
              ON c.conversation_id = r.conversation_id
             AND c.deleted = 0
            WHERE r.conversation_id = #{conversationId}
              AND r.parent_run_id IS NULL
              AND r.user_id = #{userId}
              AND c.user_id = #{userId}
            ORDER BY
              CASE WHEN r.status IN (
                'RUNNING', 'WAITING_CONFIRMATION', 'WAITING_EXTERNAL',
                'CANCEL_REQUESTED') THEN 0 ELSE 1 END,
              r.id DESC
            LIMIT 1
            """)
    AgentRun selectAuthorizedPreferredRootByConversation(
            @Param("conversationId") String conversationId,
            @Param("userId") long userId);

    @Select("""
            SELECT *
            FROM afv_agent_run FORCE INDEX (idx_agent_run_conversation_status)
            WHERE conversation_id = #{conversationId}
              AND parent_run_id IS NULL
            ORDER BY id DESC
            LIMIT 1
            """)
    AgentRun selectLatestRootByConversation(
            @Param("conversationId") String conversationId);

    @Select("""
            SELECT r.*
            FROM afv_agent_run r FORCE INDEX (idx_agent_run_user_status)
            INNER JOIN afv_agent_conversation c
              ON c.conversation_id = r.conversation_id
             AND c.deleted = 0
            WHERE r.user_id = #{userId}
              AND c.user_id = #{userId}
              AND r.parent_run_id IS NULL
              AND r.status IN (
                'RUNNING', 'WAITING_CONFIRMATION', 'WAITING_EXTERNAL',
                'CANCEL_REQUESTED')
            ORDER BY r.update_time DESC, r.id DESC
            """)
    List<AgentRun> selectAuthorizedRunningRoots(@Param("userId") long userId);

    @Select("""
            SELECT DISTINCT agent_state_session_id
            FROM afv_agent_run
            WHERE conversation_id = #{conversationId}
            ORDER BY agent_state_session_id
            """)
    List<String> selectStateSessionIdsByConversation(
            @Param("conversationId") String conversationId);

    @Select("SELECT UTC_TIMESTAMP(3)")
    LocalDateTime selectDatabaseNow();

    @Select("""
            SELECT *
            FROM afv_agent_run FORCE INDEX (uk_agent_run_parent_tool)
            WHERE parent_run_id = #{parentRunId}
              AND parent_tool_call_id = #{parentToolCallId}
            LIMIT 1
            FOR UPDATE
            """)
    AgentRun selectByParentAndToolCallForUpdate(
            @Param("parentRunId") String parentRunId,
            @Param("parentToolCallId") String parentToolCallId);

    @Select("""
            SELECT *
            FROM afv_agent_run FORCE INDEX (idx_agent_run_parent_status)
            WHERE parent_run_id = #{parentRunId}
              AND status IN (
                  'RUNNING',
                  'WAITING_CONFIRMATION',
                  'WAITING_EXTERNAL',
                  'CANCEL_REQUESTED')
            ORDER BY id
            """)
    List<AgentRun> selectActiveChildren(@Param("parentRunId") String parentRunId);

    @Update("""
            UPDATE afv_agent_run
            SET next_sequence = #{nextSequence},
                update_time = #{now}
            WHERE id = #{runId}
              AND status = 'RUNNING'
              AND owner_epoch = #{ownerEpoch}
              AND owner_instance_id IS NOT NULL
              AND lease_until > #{now}
              AND deadline_at > #{now}
            """)
    int advanceRunningSequence(
            @Param("runId") long runId,
            @Param("ownerEpoch") long ownerEpoch,
            @Param("nextSequence") long nextSequence,
            @Param("now") LocalDateTime now);

    @Update("""
            UPDATE afv_agent_run
            SET status = 'WAITING_CONFIRMATION',
                owner_instance_id = NULL,
                lease_until = NULL,
                waiting_reply_id = #{replyId},
                waiting_tool_call_id = NULL,
                waiting_tool_name = NULL,
                wait_expires_at = #{expiresAt},
                deadline_at = #{expiresAt},
                paused_through_sequence = #{pausedThroughSequence},
                next_sequence = #{nextSequence},
                update_time = #{now}
            WHERE id = #{runId}
              AND status = 'RUNNING'
              AND owner_epoch = #{ownerEpoch}
              AND owner_instance_id IS NOT NULL
              AND lease_until > #{now}
              AND deadline_at > #{now}
            """)
    int enterWaitingConfirmation(
            @Param("runId") long runId,
            @Param("ownerEpoch") long ownerEpoch,
            @Param("replyId") String replyId,
            @Param("expiresAt") LocalDateTime expiresAt,
            @Param("pausedThroughSequence") long pausedThroughSequence,
            @Param("nextSequence") long nextSequence,
            @Param("now") LocalDateTime now);

    @Update("""
            UPDATE afv_agent_run
            SET status = 'WAITING_EXTERNAL',
                owner_instance_id = NULL,
                lease_until = NULL,
                waiting_reply_id = NULL,
                waiting_tool_call_id = #{toolCallId},
                waiting_tool_name = #{toolName},
                wait_expires_at = #{expiresAt},
                paused_through_sequence = #{pausedThroughSequence},
                next_sequence = #{nextSequence},
                update_time = #{now}
            WHERE id = #{runId}
              AND status = 'RUNNING'
              AND owner_epoch = #{ownerEpoch}
              AND owner_instance_id IS NOT NULL
              AND lease_until > #{now}
              AND deadline_at > #{now}
            """)
    int enterWaitingExternal(
            @Param("runId") long runId,
            @Param("ownerEpoch") long ownerEpoch,
            @Param("toolCallId") String toolCallId,
            @Param("toolName") String toolName,
            @Param("expiresAt") LocalDateTime expiresAt,
            @Param("pausedThroughSequence") long pausedThroughSequence,
            @Param("nextSequence") long nextSequence,
            @Param("now") LocalDateTime now);

    @Update("""
            UPDATE afv_agent_run
            SET status = 'RUNNING',
                owner_instance_id = #{newOwnerInstanceId},
                owner_epoch = #{newOwnerEpoch},
                lease_until = #{leaseUntil},
                deadline_at = #{deadlineAt},
                heartbeat_at = #{now},
                waiting_reply_id = NULL,
                waiting_tool_call_id = NULL,
                waiting_tool_name = NULL,
                wait_expires_at = NULL,
                next_sequence = #{nextSequence},
                update_time = #{now}
            WHERE id = #{runId}
              AND status = 'WAITING_CONFIRMATION'
              AND owner_epoch = #{oldOwnerEpoch}
              AND waiting_reply_id = #{replyId}
              AND wait_expires_at > #{now}
            """)
    int resumeConfirmation(
            @Param("runId") long runId,
            @Param("oldOwnerEpoch") long oldOwnerEpoch,
            @Param("replyId") String replyId,
            @Param("newOwnerInstanceId") String newOwnerInstanceId,
            @Param("newOwnerEpoch") long newOwnerEpoch,
            @Param("leaseUntil") LocalDateTime leaseUntil,
            @Param("deadlineAt") LocalDateTime deadlineAt,
            @Param("nextSequence") long nextSequence,
            @Param("now") LocalDateTime now);

    @Update("""
            UPDATE afv_agent_run
            SET status = 'RUNNING',
                owner_instance_id = #{newOwnerInstanceId},
                owner_epoch = #{newOwnerEpoch},
                lease_until = #{leaseUntil},
                heartbeat_at = #{now},
                waiting_reply_id = NULL,
                waiting_tool_call_id = NULL,
                waiting_tool_name = NULL,
                wait_expires_at = NULL,
                next_sequence = #{nextSequence},
                update_time = #{now}
            WHERE id = #{runId}
              AND status = 'WAITING_EXTERNAL'
              AND owner_epoch = #{oldOwnerEpoch}
              AND waiting_tool_call_id = #{toolCallId}
              AND wait_expires_at > #{now}
              AND deadline_at > #{now}
            """)
    int resumeExternal(
            @Param("runId") long runId,
            @Param("oldOwnerEpoch") long oldOwnerEpoch,
            @Param("toolCallId") String toolCallId,
            @Param("newOwnerInstanceId") String newOwnerInstanceId,
            @Param("newOwnerEpoch") long newOwnerEpoch,
            @Param("leaseUntil") LocalDateTime leaseUntil,
            @Param("nextSequence") long nextSequence,
            @Param("now") LocalDateTime now);

    @Update("""
            UPDATE afv_agent_run
            SET heartbeat_at = UTC_TIMESTAMP(3),
                lease_until = LEAST(
                    TIMESTAMPADD(MICROSECOND, #{leaseMicros}, UTC_TIMESTAMP(3)),
                    deadline_at)
            WHERE run_id = #{runId}
              AND owner_instance_id = #{ownerInstanceId}
              AND owner_epoch = #{ownerEpoch}
              AND status = 'RUNNING'
              AND lease_until > UTC_TIMESTAMP(3)
              AND deadline_at > UTC_TIMESTAMP(3)
            """)
    int renewOwnedLease(
            @Param("runId") String runId,
            @Param("ownerInstanceId") String ownerInstanceId,
            @Param("ownerEpoch") long ownerEpoch,
            @Param("leaseMicros") long leaseMicros);

    @Select("""
            SELECT COUNT(*)
            FROM afv_agent_run FORCE INDEX (uk_agent_run_id)
            WHERE run_id = #{runId}
              AND owner_instance_id = #{ownerInstanceId}
              AND owner_epoch = #{ownerEpoch}
              AND status = 'RUNNING'
              AND lease_until > UTC_TIMESTAMP(3)
              AND deadline_at > UTC_TIMESTAMP(3)
            """)
    int countValidOwnedLease(
            @Param("runId") String runId,
            @Param("ownerInstanceId") String ownerInstanceId,
            @Param("ownerEpoch") long ownerEpoch);

    @Update("""
            UPDATE afv_agent_run
            SET cancel_acknowledged_at = UTC_TIMESTAMP(3),
                lease_until = LEAST(lease_until, UTC_TIMESTAMP(3)),
                update_time = UTC_TIMESTAMP(3)
            WHERE run_id = #{runId}
              AND owner_instance_id = #{ownerInstanceId}
              AND owner_epoch = #{ownerEpoch}
              AND status = 'CANCEL_REQUESTED'
            """)
    int acknowledgeOwnedCancellation(
            @Param("runId") String runId,
            @Param("ownerInstanceId") String ownerInstanceId,
            @Param("ownerEpoch") long ownerEpoch);

    @Update("""
            UPDATE afv_agent_run
            SET cancel_broadcast_at = #{now},
                cancel_next_attempt_at = #{nextAttemptAt},
                update_time = #{now}
            WHERE run_id = #{runId}
              AND status = 'CANCEL_REQUESTED'
            """)
    int markCancellationBroadcast(
            @Param("runId") String runId,
            @Param("now") LocalDateTime now,
            @Param("nextAttemptAt") LocalDateTime nextAttemptAt);

    @Select("""
            SELECT *
            FROM afv_agent_run FORCE INDEX (idx_agent_run_lease)
            WHERE status IN ('RUNNING', 'CANCEL_REQUESTED')
              AND lease_until IS NOT NULL
              AND lease_until <= #{now}
            ORDER BY lease_until, id
            LIMIT #{limit}
            """)
    List<AgentRun> selectExpiredLeaseCandidates(
            @Param("now") LocalDateTime now,
            @Param("limit") int limit);

    @Select("""
            SELECT *
            FROM afv_agent_run FORCE INDEX (idx_agent_run_cancel)
            WHERE status = 'CANCEL_REQUESTED'
              AND cancel_acknowledged_at IS NULL
              AND (cancel_next_attempt_at IS NULL
                   OR cancel_next_attempt_at <= #{now})
            ORDER BY cancel_next_attempt_at, id
            LIMIT #{limit}
            """)
    List<AgentRun> selectCancellationRetryCandidates(
            @Param("now") LocalDateTime now,
            @Param("limit") int limit);
}
