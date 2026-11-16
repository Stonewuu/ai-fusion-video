package com.stonewu.fusion.mapper.script;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import java.util.List;
import java.util.Map;

@Mapper
public interface ScriptSummaryMapper {
    Map<String, Object> script(@Param("id") long id);
    List<Map<String, Object>> episodes(@Param("scriptId") long scriptId);
}
