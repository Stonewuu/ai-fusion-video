package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentMessage;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface AgentMessageMapper extends BaseMapper<AgentMessage> {

    @Select("""
            SELECT MIN(message_order)
            FROM afv_agent_message
            WHERE run_id = #{runId}
              AND deleted = 0
            """)
    Long selectInitialOrderByRunId(@Param("runId") String runId);
}
