package com.stonewu.fusion.mapper.generation;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.generation.ImageTask;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface ImageTaskMapper extends BaseMapper<ImageTask> {

    Long lock(@Param("id") long id);

    ImageTask selectForUpdate(@Param("id") Long id);
}
