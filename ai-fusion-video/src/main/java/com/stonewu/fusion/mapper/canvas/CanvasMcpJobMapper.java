package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import java.time.LocalDateTime;
import java.util.List;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasMcpJobEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasMcpJobMapper extends BaseMapper<CanvasMcpJobEntity> {
    CanvasMcpJobEntity lock(@Param("id") String id);

    int insertJob(@Param("id") String id, @Param("nodeRunId") String nodeRunId, @Param("runId") String runId,
                  @Param("userId") long userId, @Param("contractId") String contractId, @Param("input") String input,
                  @Param("deadline") LocalDateTime deadline);

    List<String> next();

    int claim(@Param("owner") String owner, @Param("id") String id);

    CanvasMcpJobEntity leased(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    int heartbeat(@Param("id") String id, @Param("owner") String owner, @Param("epoch") long epoch);

    int begin(@Param("owner") String owner, @Param("epoch") long epoch, @Param("id") String id);

    int received(@Param("result") String result, @Param("hash") String hash, @Param("id") String id,
                 @Param("owner") String owner, @Param("epoch") long epoch);

    int finish(@Param("status") String status, @Param("error") String error, @Param("permit") boolean permit,
               @Param("id") String id);

    int cancel(@Param("id") String id);

    int adminCancel(@Param("message") String message, @Param("id") String id);
}
