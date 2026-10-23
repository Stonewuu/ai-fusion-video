package com.stonewu.fusion.mapper.canvas;

import java.time.LocalDateTime;
import java.util.List;

import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasTextJobEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasTextJobMapper extends BaseMapper<CanvasTextJobEntity> {
    CanvasTextJobEntity lock(@Param("id") String id);

    int insertJob(@Param("id") String id, @Param("nodeRunId") String nodeRunId, @Param("runId") String runId,
                  @Param("projectId") Long projectId, @Param("modelId") Long modelId,
                  @Param("modelLimit") Integer modelLimit, @Param("input") String input,
                  @Param("deadline") LocalDateTime deadline);

    List<String> next();

    int claim(@Param("owner") String owner, @Param("id") String id);

    CanvasTextJobEntity leased(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    int heartbeat(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    int begin(@Param("owner") String owner, @Param("epoch") long epoch, @Param("id") String id);

    int received(@Param("result") String result, @Param("hash") String hash, @Param("id") String id,
                 @Param("owner") String owner, @Param("epoch") long epoch);

    int finish(@Param("status") String status, @Param("phase") String phase, @Param("error") String error,
               @Param("permit") boolean permit, @Param("id") String id);

    int cancel(@Param("id") String id);

    int resume(@Param("id") String id);

}
