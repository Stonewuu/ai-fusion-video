package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentMessage;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;

@Mapper
public interface AgentMessageMapper extends BaseMapper<AgentMessage> {

    @Select("""
            SELECT MIN(message_order)
            FROM afv_agent_message
            WHERE run_id = #{runId}
              AND deleted = 0
            """)
    Long selectInitialOrderByRunId(@Param("runId") String runId);

    @Select("""
            SELECT *
            FROM afv_agent_message FORCE INDEX (uk_agent_message_projection_key)
            WHERE projection_key = #{projectionKey}
            LIMIT 1
            """)
    AgentMessage selectByProjectionKey(@Param("projectionKey") String projectionKey);

    @Update("""
            UPDATE afv_agent_message call_message
            LEFT JOIN afv_agent_message result_message
              ON result_message.run_id = call_message.run_id
             AND result_message.tool_call_id = call_message.tool_call_id
             AND result_message.role = 'tool'
             AND result_message.tool_status IN ('success', 'error', 'cancelled')
             AND result_message.deleted = 0
            SET call_message.tool_status = 'cancelled',
                call_message.update_time = #{updateTime}
            WHERE call_message.run_id = #{runId}
              AND call_message.role = 'tool'
              AND call_message.tool_status = 'running'
              AND call_message.deleted = 0
              AND result_message.id IS NULL
            """)
    int cancelUnfinishedTools(
            @Param("runId") String runId,
            @Param("updateTime") LocalDateTime updateTime);
}
