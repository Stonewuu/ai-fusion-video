package com.stonewu.fusion.service.project;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.entity.project.ProjectMember;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.mapper.project.ProjectMemberMapper;
import com.stonewu.fusion.service.team.TeamService;
import lombok.RequiredArgsConstructor;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * 项目服务
 */
@Service
@RequiredArgsConstructor
public class ProjectService {

    private static final int OWNER_TYPE_PERSONAL = 1;
    private static final int OWNER_TYPE_TEAM = 2;

    private final ProjectMapper projectMapper;
    private final ProjectMemberMapper memberMapper;
    private final TeamService teamService;

    @Cacheable(value = "project", key = "#id")
    public Project getById(Long id) {
        Project project = projectMapper.selectById(id);
        if (project == null) {
            throw new BusinessException("项目不存在: " + id);
        }
        return project;
    }

    public PageResult<Project> page(int pageNo, int pageSize) {
        return PageResult.of(projectMapper.selectPage(new Page<>(pageNo, pageSize), null));
    }

    @Cacheable(value = "project", key = "'owner:' + #ownerType + ':' + #ownerId")
    public List<Project> listByOwner(Integer ownerType, Long ownerId) {
        return projectMapper.selectList(new LambdaQueryWrapper<Project>()
                .eq(Project::getOwnerType, ownerType)
                .eq(Project::getOwnerId, ownerId)
                .orderByDesc(Project::getCreateTime));
    }

    public List<Project> listAccessibleByUser(Long userId) {
        Long currentTeamId = teamService.getCurrentTeamIdByUser(userId);
        if (currentTeamId == null) {
            return listByOwner(OWNER_TYPE_PERSONAL, userId);
        }
        List<Long> memberUserIds = teamService.listMemberUserIds(currentTeamId);
        return projectMapper.selectList(new LambdaQueryWrapper<Project>()
                .and(wrapper -> wrapper
                        .and(teamOwned -> teamOwned
                                .eq(Project::getOwnerType, OWNER_TYPE_TEAM)
                                .eq(Project::getOwnerId, currentTeamId))
                        .or(memberOwned -> memberOwned
                                .eq(Project::getOwnerType, OWNER_TYPE_PERSONAL)
                                .in(Project::getOwnerId, memberUserIds)))
                .orderByDesc(Project::getCreateTime));
    }

    public boolean canAccessProject(Long projectId, Long userId) {
        return canAccessProject(getById(projectId), userId);
    }

    public boolean canAccessProject(Project project, Long userId) {
        if (project == null) {
            return false;
        }
        if (OWNER_TYPE_PERSONAL == project.getOwnerType() && userId.equals(project.getOwnerId())) {
            return true;
        }
        if (isMember(project.getId(), userId)) {
            return true;
        }
        Long currentTeamId = teamService.getCurrentTeamIdByUser(userId);
        if (currentTeamId == null) {
            return false;
        }
        if (OWNER_TYPE_TEAM == project.getOwnerType() && currentTeamId.equals(project.getOwnerId())) {
            return true;
        }
        return OWNER_TYPE_PERSONAL == project.getOwnerType()
                && teamService.listMemberUserIds(currentTeamId).contains(project.getOwnerId());
    }

    // ========== 成员管理 ==========

    public boolean isMember(Long projectId, Long userId) {
        return memberMapper.exists(new LambdaQueryWrapper<ProjectMember>()
                .eq(ProjectMember::getProjectId, projectId)
                .eq(ProjectMember::getUserId, userId));
    }

    @Cacheable(value = "projectMember", key = "#projectId")
    public List<ProjectMember> listMembers(Long projectId) {
        return memberMapper.selectList(
                new LambdaQueryWrapper<ProjectMember>().eq(ProjectMember::getProjectId, projectId));
    }

}
