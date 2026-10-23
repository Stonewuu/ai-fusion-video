package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.canvas.CanvasExportBlobEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasExportBlobMapper extends BaseMapper<CanvasExportBlobEntity> {
    int reserve(@Param("id") String id, @Param("exportId") String exportId,
                @Param("storageConfigId") long storageConfigId, @Param("backend") String backend,
                @Param("key") String key, @Param("keyset") byte[] keyset, @Param("epoch") long epoch);

    int complete(@Param("bytes") long bytes, @Param("id") String id, @Param("exportId") String exportId,
                 @Param("epoch") long epoch);

    int orphan(@Param("id") String id);

    CanvasExportBlobEntity expiredOrphan(@Param("days") int days);

    long canPurge(@Param("id") String id, @Param("exportId") String exportId, @Param("days") int days);

    int purged(@Param("id") String id);
}
