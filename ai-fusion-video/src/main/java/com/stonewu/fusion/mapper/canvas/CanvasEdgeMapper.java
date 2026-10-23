package com.stonewu.fusion.mapper.canvas;

import org.apache.ibatis.annotations.Param;

import java.util.List;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasEdgeEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasEdgeMapper extends BaseMapper<CanvasEdgeEntity> {
    List<CanvasEdgeEntity> ordered(@Param("canvasId") String canvasId);

    List<CanvasEdgeEntity> deleted(@Param("canvasId") String canvasId, @Param("ids") List<String> ids);

    int deleteGraphEdge(@Param("id") String id, @Param("canvasId") String canvasId, @Param("version") long version);

    int insertGraph(@Param("row") CanvasEdgeEntity row);

    int updateGraph(@Param("row") CanvasEdgeEntity row, @Param("previousVersion") long previousVersion);
}
