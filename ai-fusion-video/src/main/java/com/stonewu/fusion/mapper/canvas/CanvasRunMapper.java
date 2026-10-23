package com.stonewu.fusion.mapper.canvas;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasRunEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface CanvasRunMapper extends BaseMapper<CanvasRunEntity> {

    CanvasRunEntity lock(@Param("id") String id);

    boolean withdrawn(@Param("scope") String scope, @Param("request") String request);

    int withdraw(@Param("scope") String scope, @Param("request") String request, @Param("canvasId") String canvasId);

    int insertAccepted(@Param("row") CanvasRunEntity row);

    int changeState(@Param("status") String status, @Param("error") String error, @Param("id") String id);
}
