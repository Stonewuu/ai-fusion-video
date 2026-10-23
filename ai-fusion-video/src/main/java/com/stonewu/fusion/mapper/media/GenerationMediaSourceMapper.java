package com.stonewu.fusion.mapper.media;

import org.apache.ibatis.annotations.Param;

import com.stonewu.fusion.service.media.model.MediaSourceType;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface GenerationMediaSourceMapper {
    record Item(long taskId, int version, int status, int taskStatus, Long projectId, Integer ownerType, Long ownerId) {
    }

    Item item(@Param("type") MediaSourceType type,
              @Param("id") long id, @Param("lock") boolean lock);

    String value(@Param("type") MediaSourceType type,
                 @Param("column") String column,
                 @Param("id") long id, @Param("version") int version);
}
