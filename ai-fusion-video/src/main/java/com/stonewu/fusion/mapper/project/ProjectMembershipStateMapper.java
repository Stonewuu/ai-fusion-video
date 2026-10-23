package com.stonewu.fusion.mapper.project;

import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.project.ProjectMembershipStateEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ProjectMembershipStateMapper extends BaseMapper<ProjectMembershipStateEntity> {
    int advance(@Param("projectId") long projectId, @Param("revision") long revision);
}
