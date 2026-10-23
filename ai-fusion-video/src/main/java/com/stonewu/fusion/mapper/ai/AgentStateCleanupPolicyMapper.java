package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentStateCleanupPolicy;
import org.apache.ibatis.annotations.Mapper;

import java.time.LocalDateTime;

@Mapper
public interface AgentStateCleanupPolicyMapper
        extends BaseMapper<AgentStateCleanupPolicy> {

    AgentStateCleanupPolicy selectCurrentForUpdate();

    LocalDateTime selectDatabaseNow();
}
