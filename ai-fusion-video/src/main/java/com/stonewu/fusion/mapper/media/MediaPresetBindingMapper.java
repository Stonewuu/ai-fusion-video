package com.stonewu.fusion.mapper.media;


import org.apache.ibatis.annotations.Param;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface MediaPresetBindingMapper {
    String readable(@Param("mediaId") String mediaId, @Param("reference") String reference);

    int publish(@Param("key") String key, @Param("id") String id, @Param("actor") long actor);
}
