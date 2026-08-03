package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentStateCleanupPolicy;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDateTime;

@Mapper
public interface AgentStateCleanupPolicyMapper
        extends BaseMapper<AgentStateCleanupPolicy> {

    @Select("""
            SELECT *
            FROM afv_agent_state_cleanup_policy
            WHERE id = 1
            FOR UPDATE
            """)
    AgentStateCleanupPolicy selectCurrentForUpdate();

    @Select("SELECT UTC_TIMESTAMP(3)")
    LocalDateTime selectDatabaseNow();
}
