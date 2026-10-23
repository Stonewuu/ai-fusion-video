package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasMcpDecisionEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasMcpDecisionMapper extends BaseMapper<CanvasMcpDecisionEntity> {
    CanvasMcpDecisionEntity lockRequest(@Param("userId") long userId, @Param("requestId") String requestId);

    int create(@Param("id") String id, @Param("jobId") String jobId, @Param("userId") long userId,
               @Param("requestId") String requestId, @Param("requestHash") String requestHash,
               @Param("action") String action, @Param("note") String note);
}
