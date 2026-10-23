package com.stonewu.fusion.mapper.canvas;


import java.util.List;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasExportEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasExportMapper extends BaseMapper<CanvasExportEntity> {
    CanvasExportEntity lock(@Param("id") String id);

    int create(@Param("row") CanvasExportEntity row);

    List<String> pending();

    int claim(@Param("worker") String worker, @Param("id") String id);

    int heartbeat(@Param("id") String id, @Param("worker") String worker, @Param("epoch") long epoch);

    int complete(@Param("blobId") String blobId, @Param("bytes") long bytes, @Param("hash") String hash,
                 @Param("id") String id);

    int cancel(@Param("id") String id);

    int finish(@Param("status") String status, @Param("error") String error, @Param("id") String id);

    int event(@Param("actor") Long actor, @Param("type") String type, @Param("detail") String detail,
              @Param("id") String id);

    int activeNodeRun(@Param("id") String id);
}
