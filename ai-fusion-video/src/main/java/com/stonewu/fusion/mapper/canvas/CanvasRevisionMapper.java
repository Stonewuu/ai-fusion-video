package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import java.util.List;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasRevisionEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasRevisionMapper extends BaseMapper<CanvasRevisionEntity> {
    List<CanvasRevisionEntity> history(@Param("canvasId") String canvasId, @Param("beforeRevision") long beforeRevision,
                                       @Param("limit") int limit);

    CanvasRevisionEntity detail(@Param("canvasId") String canvasId, @Param("id") String id);

    int create(@Param("id") String id, @Param("canvasId") String canvasId, @Param("title") String title,
               @Param("revision") long revision, @Param("sequence") long sequence, @Param("snapshot") String snapshot,
               @Param("actor") long actor);
}
