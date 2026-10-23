package com.stonewu.fusion.mapper.script;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.script.Script;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface ScriptMapper extends BaseMapper<Script> {
    Long lockLatest(@Param("projectId") long projectId);
}
