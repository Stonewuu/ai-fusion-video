package com.stonewu.fusion.mapper.media;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.media.MediaBlobCleanupEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface MediaBlobCleanupMapper extends BaseMapper<MediaBlobCleanupEntity> {
    record Lease(String ownerId, long epoch, String status, Boolean valid) {
    }

    int discover(@Param("days") int days);

    String next();

    Boolean oldEnough(@Param("days") int days, @Param("id") String id);

    int claim(@Param("owner") String owner, @Param("id") String id);

    int purged(@Param("id") String id);

    int defer(@Param("minutes") int minutes, @Param("id") String id);

    int block(@Param("reason") String reason, @Param("id") String id);

    Lease lockLease(@Param("id") String id);
}
