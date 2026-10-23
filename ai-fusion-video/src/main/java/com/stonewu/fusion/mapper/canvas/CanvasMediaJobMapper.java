package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasMediaJobEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasMediaJobMapper extends BaseMapper<CanvasMediaJobEntity> {
    int acquirePermit(@Param("id") String id);

    int pollResult(@Param("id") String id, @Param("failed") boolean failed);
}
