package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import java.util.List;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasCommentEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasCommentMapper extends BaseMapper<CanvasCommentEntity> {
    List<CanvasCommentEntity> history(@Param("canvasId") String canvasId, @Param("limit") int limit,
                                      @Param("offset") int offset);

    CanvasCommentEntity detail(@Param("canvasId") String canvasId, @Param("id") String id);

    int create(@Param("id") String id, @Param("canvasId") String canvasId, @Param("nodeId") String nodeId,
               @Param("parentId") String parentId, @Param("resourceVersion") Long resourceVersion,
               @Param("content") String content, @Param("actor") long actor);

    int patch(@Param("content") String content, @Param("resolved") boolean resolved, @Param("deleted") boolean deleted,
              @Param("canvasId") String canvasId, @Param("id") String id, @Param("version") long version);
}
