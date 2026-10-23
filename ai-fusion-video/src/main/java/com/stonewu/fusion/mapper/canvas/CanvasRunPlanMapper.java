package com.stonewu.fusion.mapper.canvas;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasRunPlanEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface CanvasRunPlanMapper extends BaseMapper<CanvasRunPlanEntity> {

    int store(@Param("row") CanvasRunPlanEntity row);
}
