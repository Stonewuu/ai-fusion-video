package com.stonewu.fusion.mapper.media;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;

import java.util.List;

import org.apache.ibatis.annotations.Param;

import com.stonewu.fusion.entity.media.MediaJobSubmissionEntity;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface MediaJobSubmissionMapper extends BaseMapper<MediaJobSubmissionEntity> {

    int prepare(@Param("id") String id, @Param("step") String step, @Param("attempt") String attempt);

    int acknowledge(@Param("providerId") String providerId, @Param("attempt") String attempt);

    List<String> lockAttempts(@Param("id") String id);

}
