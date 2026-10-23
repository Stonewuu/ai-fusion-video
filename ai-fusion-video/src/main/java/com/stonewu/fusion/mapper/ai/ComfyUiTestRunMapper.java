package com.stonewu.fusion.mapper.ai;

import java.util.List;

import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.ai.ComfyUiTestRunEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ComfyUiTestRunMapper extends BaseMapper<ComfyUiTestRunEntity> {
    int begin(@Param("promptId") String promptId, @Param("request") String request, @Param("id") String id);

    List<String> lockUnacknowledged(@Param("id") String id, @Param("promptId") String promptId);

    int acknowledge(@Param("id") String id);

    int polled(@Param("error") String error, @Param("id") String id);

    List<String> history(@Param("versionId") long versionId, @Param("actor") long actor, @Param("team") Long team);

    ComfyUiTestRunEntity lock(@Param("id") String id);
}
