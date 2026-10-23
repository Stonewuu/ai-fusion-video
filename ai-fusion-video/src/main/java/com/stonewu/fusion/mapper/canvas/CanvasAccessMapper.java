package com.stonewu.fusion.mapper.canvas;


import org.apache.ibatis.annotations.Param;

import java.util.List;

import com.stonewu.fusion.repository.canvas.CanvasAccessRepository.ProjectScope;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface CanvasAccessMapper {
    int activeTeamMember(@Param("userId") long userId, @Param("teamId") long teamId);

    ProjectScope project(@Param("projectId") long projectId, @Param("userId") long userId,
                         @Param("teamId") Long teamId);

    List<Long> readableProjects(@Param("userId") long userId, @Param("teamId") Long teamId);

    int canEditProject(@Param("projectId") long projectId, @Param("userId") long userId, @Param("teamId") Long teamId);

    int referenceBelongs(@Param("type") CanvasValueType type, @Param("id") long id, @Param("projectId") long projectId);
}
