package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import java.util.List;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasMediaImportEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasMediaImportMapper extends BaseMapper<CanvasMediaImportEntity> {
    List<String> request(@Param("actor") long actor, @Param("requestId") String requestId,
                         @Param("canvasId") String canvasId, @Param("teamId") Long teamId);

    List<String> history(@Param("canvasId") String canvasId, @Param("actor") long actor, @Param("teamId") Long teamId,
                         @Param("offset") int offset);

    List<String> active(@Param("canvasId") String canvasId, @Param("actor") long actor, @Param("teamId") Long teamId);

    int insertImport(@Param("id") String id, @Param("canvasId") String canvasId, @Param("teamId") Long teamId,
                     @Param("placement") String placement);
}
