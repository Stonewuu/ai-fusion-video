package com.stonewu.fusion.mapper.canvas;

import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasValueSnapshotEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasValueSnapshotMapper extends BaseMapper<CanvasValueSnapshotEntity> {
    int create(@Param("row") CanvasValueSnapshotEntity row);
}
