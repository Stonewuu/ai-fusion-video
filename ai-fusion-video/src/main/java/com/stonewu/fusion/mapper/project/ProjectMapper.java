package com.stonewu.fusion.mapper.project;


import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.project.Project;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface ProjectMapper extends BaseMapper<Project> {

    Long lock(@Param("id") long id);

    Long lockActive(@Param("id") long id);
}
