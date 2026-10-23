package com.stonewu.fusion.mapper.media;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.media.MediaPreviewJobEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface MediaPreviewJobMapper extends BaseMapper<MediaPreviewJobEntity> {
    String current(@Param("sourceId") String sourceId, @Param("version") int version, @Param("profile") String profile,
                   @Param("lock") boolean lock);

    Source source(@Param("jobId") String jobId, @Param("profile") String profile);

    record Source(String id, int version, Long projectId) {
    }

    int setCurrent(@Param("sourceId") String sourceId, @Param("version") int version, @Param("profile") String profile,
                   @Param("jobId") String jobId);
}
