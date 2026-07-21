package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentRun;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

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
}
