package com.stonewu.fusion.mapper.project;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.stonewu.fusion.entity.project.ProjectMember;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

@Mapper
public interface ProjectMemberMapper extends BaseMapper<ProjectMember> {
    List<Map<String, Object>> lockProjectMembers(@Param("projectId") long projectId);

    int createOwner(@Param("projectId") long projectId, @Param("userId") long userId);

    record PersonRow(long id, String username, String name, int status, boolean deleted, Integer role) {
    }

    IPage<PersonRow> members(Page<PersonRow> page, @Param("projectId") long projectId,
                             @Param("ownerId") long ownerId, @Param("ownerType") int ownerType);

    PersonRow findPerson(@Param("ownerId") long ownerId, @Param("ownerType") int ownerType,
                         @Param("projectId") long projectId, @Param("username") String username);

    int removeMember(@Param("projectId") long projectId, @Param("userId") long userId);

    int removeProjectMembers(@Param("projectId") long projectId);

    int grantMember(@Param("projectId") long projectId, @Param("userId") long userId, @Param("role") Integer role);
}
