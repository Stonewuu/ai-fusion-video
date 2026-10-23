package com.stonewu.fusion.mapper.project;


import org.apache.ibatis.annotations.Param;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.project.ProjectMembershipCommandEntity;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface ProjectMembershipCommandMapper extends BaseMapper<ProjectMembershipCommandEntity> {
    IPage<ProjectMembershipCommandEntity> history(Page<ProjectMembershipCommandEntity> page,
                                                  @Param("projectId") long projectId);

    int create(@Param("row") ProjectMembershipCommandEntity row);

    ProjectMembershipCommandEntity nextUnpublished();
}
