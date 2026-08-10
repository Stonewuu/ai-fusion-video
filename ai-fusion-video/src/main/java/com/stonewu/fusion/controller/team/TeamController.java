package com.stonewu.fusion.controller.team;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.stonewu.fusion.common.CommonResult;
import com.stonewu.fusion.common.PageResult;
import com.stonewu.fusion.controller.team.vo.*;
import com.stonewu.fusion.convert.team.TeamConvert;
import com.stonewu.fusion.entity.team.Team;
import com.stonewu.fusion.entity.team.TeamMember;
import com.stonewu.fusion.mapper.team.TeamMemberMapper;
import com.stonewu.fusion.service.team.TeamService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;

import java.util.List;

import static com.stonewu.fusion.common.CommonResult.success;
import static com.stonewu.fusion.security.SecurityUtils.getCurrentUserId;

@Tag(name = "团队管理")
@RestController
@RequestMapping("/api/team")
@RequiredArgsConstructor
public class TeamController {

    private final TeamService teamService;
    private final TeamMemberMapper teamMemberMapper;

    @PostMapping("/create")
    @Operation(summary = "创建团队")
    public CommonResult<Long> createTeam(@Valid @RequestBody TeamCreateReqVO reqVO) {
        // ownerUserId 由后端从 token 获取
        Team team = teamService.createTeam(reqVO.getName(), reqVO.getDescription(), getCurrentUserId());
        return success(team.getId());
    }

    @GetMapping("/get")
    @Operation(summary = "获取团队详情")
    @Parameter(name = "id", description = "团队ID", required = true)
    public CommonResult<TeamRespVO> getTeam(@RequestParam("id") Long id) {
        Team team = teamService.getById(id);
        return success(team == null ? null : enrichTeamVO(team));
    }

    @GetMapping("/page")
    @Operation(summary = "团队分页列表")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<PageResult<TeamRespVO>> getTeamPage(@Valid TeamPageReqVO reqVO) {
        return success(teamService.getPage(reqVO.getName(), reqVO.getStatus(),
                reqVO.getPageNo(), reqVO.getPageSize())
                .map(this::enrichTeamVO));
    }

    @PutMapping("/update")
    @Operation(summary = "更新团队")
    public CommonResult<Boolean> updateTeam(@Valid @RequestBody TeamUpdateReqVO reqVO) {
        teamService.updateTeam(reqVO.getId(), reqVO.getName(), reqVO.getDescription(), reqVO.getLogo(), reqVO.getStatus());
        return success(true);
    }

    @DeleteMapping("/delete")
    @Operation(summary = "删除团队")
    @PreAuthorize("hasRole('ADMIN')")
    public CommonResult<Boolean> deleteTeam(@RequestParam("id") Long id) {
        teamService.deleteTeam(id);
        return success(true);
    }

    // ==================== 成员管理 ====================

    @PostMapping("/member/add")
    @Operation(summary = "添加团队成员")
    public CommonResult<Long> addMember(@Valid @RequestBody TeamMemberAddReqVO reqVO) {
        return success(teamService.addMember(reqVO.getTeamId(), reqVO.getUserId(), reqVO.getRole()));
    }

    @PostMapping("/member/remove")
    @Operation(summary = "移除团队成员")
    public CommonResult<Boolean> removeMember(@RequestParam("teamId") Long teamId, @RequestParam("userId") Long userId) {
        teamService.removeMember(teamId, userId);
        return success(true);
    }

    @PostMapping("/member/change-role")
    @Operation(summary = "变更成员角色")
    public CommonResult<Boolean> changeMemberRole(@RequestParam("teamId") Long teamId,
                                                   @RequestParam("userId") Long userId,
                                                   @RequestParam("role") Integer role) {
        teamService.changeMemberRole(teamId, userId, role);
        return success(true);
    }

    @GetMapping("/member/list")
    @Operation(summary = "获取团队成员列表")
    public CommonResult<List<TeamMemberRespVO>> getMemberList(@RequestParam("teamId") Long teamId) {
        return success(TeamConvert.INSTANCE.convertMemberList(teamService.getMemberList(teamId)));
    }

    @PutMapping("/current")
    @Operation(summary = "切换当前团队")
    public CommonResult<Boolean> switchCurrentTeam(@RequestParam("teamId") Long teamId, HttpServletRequest request) {
        teamService.switchCurrentTeam(getCurrentUserId(), teamId, getTokenFromRequest(request));
        return success(true);
    }

    /**
     * MapStruct 自动映射基础字段 + 手动补充 memberCount
     */
    private TeamRespVO enrichTeamVO(Team team) {
        TeamRespVO vo = TeamConvert.INSTANCE.convert(team);
        vo.setMemberCount(teamMemberMapper.selectCount(
                new LambdaQueryWrapper<TeamMember>().eq(TeamMember::getTeamId, team.getId())));
        return vo;
    }

    private String getTokenFromRequest(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (StringUtils.hasText(bearerToken)) {
            if (bearerToken.startsWith("Bearer ")) {
                return bearerToken.substring(7);
            }
            return bearerToken;
        }
        return request.getParameter("access_token");
    }
}
