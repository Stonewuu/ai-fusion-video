package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import com.stonewu.fusion.entity.canvas.CanvasMediaImportWithdrawalEntity;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasMediaImportWithdrawalMapper {
    CanvasMediaImportWithdrawalEntity get(@Param("actor") long actor, @Param("requestId") String requestId,
                                          @Param("lock") boolean lock);

    int insert(@Param("actor") long actor, @Param("requestId") String requestId, @Param("canvasId") String canvasId,
               @Param("teamId") Long teamId);
}
