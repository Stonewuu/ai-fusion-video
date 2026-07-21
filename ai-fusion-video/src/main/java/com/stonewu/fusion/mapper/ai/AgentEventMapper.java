package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentEvent;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

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
}
