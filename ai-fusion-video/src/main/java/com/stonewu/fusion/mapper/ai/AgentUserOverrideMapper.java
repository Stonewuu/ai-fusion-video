package com.stonewu.fusion.mapper.ai;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;

import com.stonewu.fusion.entity.ai.AgentUserOverrideEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface AgentUserOverrideMapper extends BaseMapper<AgentUserOverrideEntity> {

    int updateRevision(@Param("json") String json, @Param("userId") long userId, @Param("type") String type,
                       @Param("revision") long revision);
}
