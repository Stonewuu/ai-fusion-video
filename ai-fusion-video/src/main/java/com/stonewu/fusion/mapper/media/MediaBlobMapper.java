package com.stonewu.fusion.mapper.media;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.media.MediaBlobEntity;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface MediaBlobMapper extends BaseMapper<MediaBlobEntity> {
    String lockState(@Param("id") String id);

    long countStorageReferences(@Param("id") Long id);

    String lockReady(@Param("id") String id);

    int orphanStaging(@Param("id") String id);

    int orphan(@Param("id") String id);
}
