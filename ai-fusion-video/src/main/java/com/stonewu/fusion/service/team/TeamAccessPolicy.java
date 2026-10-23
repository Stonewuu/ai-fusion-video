package com.stonewu.fusion.service.team;

import com.baomidou.mybatisplus.core.toolkit.Wrappers;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.team.Team;
import com.stonewu.fusion.entity.team.TeamMember;
import com.stonewu.fusion.enums.team.TeamMemberRoleEnum;
import com.stonewu.fusion.mapper.team.TeamMapper;
import com.stonewu.fusion.mapper.team.TeamMemberMapper;
import com.stonewu.fusion.security.SecurityUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.Objects;

/** 团队授权实时读取，写操作在团队行锁内校验，不能缓存权限结果。 */
@Component
@RequiredArgsConstructor
public class TeamAccessPolicy {
    private final TeamMapper teams;
    private final TeamMemberMapper members;

    public Team requireRead(long teamId) {
        long userId = SecurityUtils.requireCurrentUserId();
        Team team = requireTeam(teams.selectById(teamId));
        if (systemAdmin() || Objects.equals(team.getOwnerUserId(), userId) || member(teamId, userId) != null) {
            return team;
        }
        throw denied();
    }

    public Team requireManage(long teamId) {
        long userId = SecurityUtils.requireCurrentUserId();
        Team team = requireTeam(teams.lock(teamId));
        if (systemAdmin() || Objects.equals(team.getOwnerUserId(), userId)) {
            return team;
        }
        TeamMember member = member(teamId, userId);
        if (member != null && Objects.equals(member.getRole(), TeamMemberRoleEnum.ADMIN.getRole())) {
            return team;
        }
        throw denied();
    }

    public void requireOwner(Team team) {
        if (!systemAdmin() && !Objects.equals(team.getOwnerUserId(), SecurityUtils.requireCurrentUserId())) {
            throw denied();
        }
    }

    private TeamMember member(long teamId, long userId) {
        return members.selectOne(Wrappers.<TeamMember>lambdaQuery()
                .eq(TeamMember::getTeamId, teamId).eq(TeamMember::getUserId, userId)
                .eq(TeamMember::getStatus, 1));
    }

    private boolean systemAdmin() {
        return SecurityUtils.getCurrentUserDetails().getAuthorities().stream()
                .anyMatch(authority -> "ROLE_ADMIN".equals(authority.getAuthority()));
    }

    private Team requireTeam(Team team) {
        if (team == null) {
            throw new BusinessException(404, "团队不存在");
        }
        return team;
    }

    private BusinessException denied() {
        return new BusinessException(403, "无权访问或管理此团队");
    }
}
