package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentWorkspaceConfig;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface AgentWorkspaceConfigMapper extends BaseMapper<AgentWorkspaceConfig> {

    AgentWorkspaceConfig selectCurrentForUpdate();
}
