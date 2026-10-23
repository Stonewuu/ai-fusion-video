package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import java.util.List;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasNodeEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasNodeMapper extends BaseMapper<CanvasNodeEntity> {
    List<CanvasNodeEntity> ordered(@Param("canvasId") String canvasId);

    List<CanvasNodeEntity> deleted(@Param("canvasId") String canvasId, @Param("ids") List<String> ids);

    int deleteGraphNode(@Param("id") String id, @Param("canvasId") String canvasId,
                        @Param("configVersion") long configVersion, @Param("layoutVersion") long layoutVersion);

    int insertGraph(@Param("row") CanvasNodeEntity row);

    int updateGraph(@Param("row") CanvasNodeEntity row, @Param("previousConfig") long previousConfig,
                    @Param("previousLayout") long previousLayout);

    int restoreGraph(@Param("row") CanvasNodeEntity row, @Param("previousConfig") long previousConfig,
                     @Param("previousLayout") long previousLayout);
}
