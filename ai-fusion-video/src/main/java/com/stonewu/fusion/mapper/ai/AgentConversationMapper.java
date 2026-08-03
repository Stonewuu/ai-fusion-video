package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentConversation;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;
import java.util.List;

@Mapper
public interface AgentConversationMapper extends BaseMapper<AgentConversation> {

    @Select("""
            SELECT *
            FROM afv_agent_conversation
            WHERE conversation_id = #{conversationId}
              AND deleted = 0
            LIMIT 1
            FOR UPDATE
            """)
    AgentConversation selectByConversationIdForUpdate(
            @Param("conversationId") String conversationId);

    @Select("SELECT UTC_TIMESTAMP(3)")
    LocalDateTime selectDatabaseNow();

    @Select("""
            SELECT conversation.*
            FROM afv_agent_conversation conversation
                FORCE INDEX (idx_agent_conversation_state_retention)
            WHERE conversation.deleted = 0
              AND conversation.agent_state_expired_at IS NULL
              AND conversation.agent_state_last_active_at IS NOT NULL
              AND conversation.agent_state_last_active_at <= #{cutoff}
              AND NOT EXISTS (
                  SELECT 1
                  FROM afv_agent_run active_run
                  WHERE active_run.conversation_id = conversation.conversation_id
                    AND active_run.parent_run_id IS NULL
                    AND active_run.status IN (
                        'RUNNING',
                        'WAITING_CONFIRMATION',
                        'WAITING_EXTERNAL',
                        'CANCEL_REQUESTED'
                    )
              )
            ORDER BY conversation.agent_state_last_active_at, conversation.id
            LIMIT #{limit}
            """)
    List<AgentConversation> selectAgentStateCleanupCandidates(
            @Param("cutoff") LocalDateTime cutoff,
            @Param("limit") int limit);

    @Update("""
            UPDATE afv_agent_conversation conversation
            SET conversation.agent_state_expired_at = #{expiredAt},
                conversation.update_time = conversation.update_time
            WHERE conversation.id = #{id}
              AND conversation.deleted = 0
              AND conversation.agent_state_expired_at IS NULL
              AND conversation.agent_state_last_active_at = #{expectedLastActiveAt}
              AND NOT EXISTS (
                  SELECT 1
                  FROM afv_agent_run active_run
                  WHERE active_run.conversation_id = conversation.conversation_id
                    AND active_run.parent_run_id IS NULL
                    AND active_run.status IN (
                        'RUNNING',
                        'WAITING_CONFIRMATION',
                        'WAITING_EXTERNAL',
                        'CANCEL_REQUESTED'
                    )
              )
            """)
    int markAgentStateExpired(
            @Param("id") long id,
            @Param("expectedLastActiveAt") LocalDateTime expectedLastActiveAt,
            @Param("expiredAt") LocalDateTime expiredAt);
}
