package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import java.util.List;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasOperationEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasOperationMapper extends BaseMapper<CanvasOperationEntity> {
    CanvasOperationEntity undoSource(@Param("canvasId") String canvasId, @Param("commandId") String commandId);

    List<CanvasOperationEntity> history(@Param("canvasId") String canvasId,
                                        @Param("beforeRevision") long beforeRevision, @Param("limit") int limit);

    int append(@Param("row") CanvasOperationEntity row);
}
