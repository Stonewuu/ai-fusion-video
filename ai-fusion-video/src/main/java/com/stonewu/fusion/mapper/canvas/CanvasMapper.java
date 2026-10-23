package com.stonewu.fusion.mapper.canvas;

import java.util.List;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasMapper extends BaseMapper<CanvasEntity> {
    CanvasEntity lock(@Param("id") String id);

    int admit(@Param("id") String id, @Param("title") String title, @Param("projectId") Long projectId,
              @Param("ownerType") int ownerType, @Param("ownerId") long ownerId, @Param("userId") long userId);

    int advance(@Param("id") String id, @Param("revision") long revision, @Param("sequence") long sequence);

    int countProjectManagers(@Param("projectId") long projectId, @Param("userId") long userId);

    IPage<CanvasEntity> accessible(Page<CanvasEntity> page, @Param("userId") long userId,
                                   @Param("projects") List<Long> projects, @Param("projectId") Long projectId,
                                   @Param("search") String search, @Param("archived") boolean archived);
}
