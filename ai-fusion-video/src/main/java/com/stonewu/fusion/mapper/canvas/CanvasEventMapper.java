package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasEventEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasEventMapper extends BaseMapper<CanvasEventEntity> {
    int append(@Param("row") CanvasEventEntity row);

    Long firstSequence(@Param("canvasId") String canvasId);

    int membershipChanged(@Param("id") String id, @Param("sequence") long sequence, @Param("payload") String payload);
}
