package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.AgentWorkspaceConfig;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

@Mapper
public interface AgentWorkspaceConfigMapper extends BaseMapper<AgentWorkspaceConfig> {

    @Select("SELECT * FROM afv_agent_workspace_config WHERE id = 1 AND deleted = 0 FOR UPDATE")
    AgentWorkspaceConfig selectCurrentForUpdate();
}
