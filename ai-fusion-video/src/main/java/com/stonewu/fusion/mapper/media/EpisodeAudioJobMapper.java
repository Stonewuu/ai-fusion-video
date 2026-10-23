package com.stonewu.fusion.mapper.media;

import java.util.List;

import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.media.EpisodeAudioJobEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface EpisodeAudioJobMapper extends BaseMapper<EpisodeAudioJobEntity> {
    EpisodeAudioJobEntity lock(@Param("id") String id);

    List<String> unresolved(@Param("id") String id);

    String successfulResult(@Param("id") String id);

    int pollResult(@Param("id") String id, @Param("failed") boolean failed);

}
