package com.stonewu.fusion.service.team;


import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.entity.team.Team;
import com.stonewu.fusion.entity.team.TeamMember;
import com.stonewu.fusion.enums.team.TeamMemberRoleEnum;
import com.stonewu.fusion.mapper.team.TeamMapper;
import com.stonewu.fusion.mapper.team.TeamMemberMapper;
import com.stonewu.fusion.mapper.system.UserMapper;
import com.stonewu.fusion.security.SecurityUtils;
import com.stonewu.fusion.security.TokenService;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.annotation.Isolation;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class TeamService {

    private static final int OWNER_TYPE_TEAM = 2;

    private final TeamMapper teamMapper;
    private final TeamMemberMapper teamMemberMapper;
    private final TokenService tokenService;
    private final TeamAccessPolicy access;
    private final UserMapper users;

    @Data
    @AllArgsConstructor
    public static class OwnerScope {
        private Integer ownerType;
        private Long ownerId;
    }

    @Transactional
    public Team createTeam(String name, String description, Long ownerUserId) {
        if (getSingleTeam() != null) {
            throw new BusinessException(400, "开源版仅支持单团队");
        }
        return createTeamRecord(name, description, ownerUserId, TeamMemberRoleEnum.OWNER.getRole());
    }

    @Transactional
    public Team createInitialTeam(String name, Long ownerUserId) {
        if (getSingleTeam() != null) {
            throw new BusinessException(400, "默认团队已存在");
        }
        return createTeamRecord(name, null, ownerUserId, TeamMemberRoleEnum.OWNER.getRole());
    }

    public Team getSingleTeam() {
        return teamMapper.selectPage(new Page<>(1, 1, false), new LambdaQueryWrapper<Team>()
                .orderByAsc(Team::getId)).getRecords().stream().findFirst().orElse(null);
    }

    public Team getRequiredSingleTeam() {
        Team team = getSingleTeam();
        if (team == null) {
            throw new BusinessException(500, "默认团队不存在，请先完成管理员初始化");
        }
        return team;
    }

    public Long getCurrentTeamIdByUser(Long userId) {
        Long currentUserId = SecurityUtils.getCurrentUserId();
        Long authenticatedTeamId = SecurityUtils.getCurrentTeamId();
        if (currentUserId != null && currentUserId.equals(userId)
                && authenticatedTeamId != null
                && isUserInTeam(authenticatedTeamId, userId)) {
            return authenticatedTeamId;
        }
        TeamMember member = teamMemberMapper.selectPage(new Page<>(1, 1, false), new LambdaQueryWrapper<TeamMember>()
                .eq(TeamMember::getUserId, userId)
                .eq(TeamMember::getStatus, 1)
                .orderByAsc(TeamMember::getTeamId)
                .orderByAsc(TeamMember::getId)).getRecords().stream().findFirst().orElse(null);
        return member != null ? member.getTeamId() : null;
    }

    public Long getRequiredCurrentTeamIdByUser(Long userId) {
        Long teamId = getCurrentTeamIdByUser(userId);
        if (teamId == null) {
            throw new BusinessException(404, "当前用户未加入任何团队");
        }
        return teamId;
    }

    public OwnerScope getRequiredCurrentOwnerScopeByUser(Long userId) {
        return new OwnerScope(OWNER_TYPE_TEAM, getRequiredCurrentTeamIdByUser(userId));
    }

    @Transactional
    public void switchCurrentTeam(Long userId, Long teamId, String accessToken) {
        if (!isUserInTeam(teamId, userId)) {
            throw new BusinessException(403, "无权切换到该团队");
        }
        boolean updated = tokenService.updateCurrentTeam(accessToken, userId, teamId);
        if (!updated) {
            throw new BusinessException(401, "登录态已失效，请重新登录");
        }
        SecurityUtils.refreshCurrentTeamId(teamId);
    }

    public boolean isUserInTeam(Long teamId, Long userId) {
        if (teamId == null) {
            return false;
        }
        return teamMemberMapper.exists(new LambdaQueryWrapper<TeamMember>()
                .eq(TeamMember::getTeamId, teamId)
                .eq(TeamMember::getUserId, userId)
                .eq(TeamMember::getStatus, 1));
    }

    public List<Long> listMemberUserIds(Long teamId) {
        if (teamId == null) {
            return List.of();
        }
        return teamMemberMapper.selectList(new LambdaQueryWrapper<TeamMember>()
                        .eq(TeamMember::getTeamId, teamId)
                        .eq(TeamMember::getStatus, 1))
                .stream()
                .map(TeamMember::getUserId)
                .distinct()
                .collect(Collectors.toList());
    }

    public boolean isUserInSingleTeam(Long userId) {
        Team team = getSingleTeam();
        if (team == null) {
            return false;
        }
        return teamMemberMapper.exists(new LambdaQueryWrapper<TeamMember>()
                .eq(TeamMember::getTeamId, team.getId())
                .eq(TeamMember::getUserId, userId));
    }

    public List<Long> listSingleTeamMemberUserIds() {
        return listMemberUserIds(getRequiredSingleTeam().getId());
    }

    @Transactional
    public void addRegisteredUserToSingleTeam(Long userId) {
        Team team = getRequiredSingleTeam();
        addMemberRecord(team.getId(), userId, TeamMemberRoleEnum.MEMBER.getRole());
    }

    public Team getById(Long id) {
        return access.requireRead(id);
    }

    public PageResult<Team> getPage(String name, Integer status, int pageNo, int pageSize) {
        LambdaQueryWrapper<Team> wrapper = new LambdaQueryWrapper<>();
        wrapper.like(name != null, Team::getName, name)
                .eq(status != null, Team::getStatus, status)
                .orderByDesc(Team::getId);
        return PageResult.of(teamMapper.selectPage(new Page<>(pageNo, pageSize), wrapper));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void updateTeam(Long id, String name, String description, String logo, Integer status) {
        Team team = access.requireManage(id);
        if (name != null) {
            team.setName(name);
        }
        if (description != null) {
            team.setDescription(description);
        }
        if (logo != null) {
            team.setLogo(logo);
        }
        if (status != null) {
            team.setStatus(status);
        }
        teamMapper.updateById(team);
    }

    @Transactional
    public void deleteTeam(Long id) {
        throw new BusinessException(400, "开源版仅支持单团队，不支持删除团队");
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public Long addMember(Long teamId, Long userId, Integer role) {
        Team team = access.requireManage(teamId);
        int assignedRole = role == null ? TeamMemberRoleEnum.MEMBER.getRole() : role;
        validateRole(assignedRole);
        protectOwner(team, userId);
        if (assignedRole == TeamMemberRoleEnum.ADMIN.getRole()) {
            access.requireOwner(team);
        }
        var user = users.selectById(userId);
        if (user == null || !Objects.equals(user.getStatus(), 1)) {
            throw new BusinessException(400, "只能添加有效用户");
        }
        return addMemberRecord(teamId, userId, assignedRole);
    }

    private Long addMemberRecord(Long teamId, Long userId, Integer role) {
        var identity = new LambdaQueryWrapper<TeamMember>()
                .eq(TeamMember::getTeamId, teamId)
                .eq(TeamMember::getUserId, userId);
        boolean exists = teamMemberMapper.exists(identity);
        if (exists) {
            throw new BusinessException(400, "该用户已是团队成员");
        }
        TeamMember member = TeamMember.builder()
                .teamId(teamId).userId(userId)
                .role(role)
                .status(1).joinTime(LocalDateTime.now())
                .build();
        // 唯一键覆盖软删除记录，重新邀请必须恢复原记录并采用本次明确授予的角色。
        if (teamMemberMapper.restoreRemoved(member) == 1) {
            return teamMemberMapper.selectOne(identity).getId();
        }
        teamMemberMapper.insert(member);
        return member.getId();
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void removeMember(Long teamId, Long userId) {
        Team team = access.requireManage(teamId);
        TeamMember member = requireMember(teamId, userId);
        protectOwner(team, userId);
        if (!Objects.equals(member.getRole(), TeamMemberRoleEnum.MEMBER.getRole())) {
            access.requireOwner(team);
        }
        teamMemberMapper.delete(new LambdaQueryWrapper<TeamMember>()
                .eq(TeamMember::getTeamId, teamId)
                .eq(TeamMember::getUserId, userId));
    }

    @Transactional(isolation = Isolation.READ_COMMITTED)
    public void changeMemberRole(Long teamId, Long userId, Integer role) {
        Team team = access.requireManage(teamId);
        access.requireOwner(team);
        protectOwner(team, userId);
        validateRole(role);
        TeamMember member = requireMember(teamId, userId);
        member.setRole(role);
        teamMemberMapper.updateById(member);
    }

    private TeamMember requireMember(Long teamId, Long userId) {
        TeamMember member = teamMemberMapper.selectOne(new LambdaQueryWrapper<TeamMember>()
                .eq(TeamMember::getTeamId, teamId)
                .eq(TeamMember::getUserId, userId));
        if (member == null) {
            throw new BusinessException(404, "成员不存在");
        }
        return member;
    }

    public List<TeamMember> getMemberList(Long teamId) {
        access.requireRead(teamId);
        return teamMemberMapper.selectList(new LambdaQueryWrapper<TeamMember>().eq(TeamMember::getTeamId, teamId));
    }

    private void protectOwner(Team team, Long userId) {
        if (Objects.equals(team.getOwnerUserId(), userId)) {
            throw new BusinessException(400, "不能移除团队创建者或修改其角色");
        }
    }

    private void validateRole(Integer role) {
        if (!Objects.equals(role, TeamMemberRoleEnum.ADMIN.getRole())
                && !Objects.equals(role, TeamMemberRoleEnum.MEMBER.getRole())) {
            throw new BusinessException(400, "成员角色只能为管理员或普通成员");
        }
    }

    private Team createTeamRecord(String name, String description, Long ownerUserId, Integer ownerRole) {
        Team team = Team.builder()
                .name(name)
                .description(description)
                .ownerUserId(ownerUserId)
                .status(1)
                .build();
        teamMapper.insert(team);
        teamMemberMapper.insert(TeamMember.builder()
                .teamId(team.getId())
                .userId(ownerUserId)
                .role(ownerRole)
                .status(1)
                .joinTime(LocalDateTime.now())
                .build());
        return team;
    }
}
