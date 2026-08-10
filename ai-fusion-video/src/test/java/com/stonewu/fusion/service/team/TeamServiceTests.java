package com.stonewu.fusion.service.team;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.team.TeamMember;
import com.stonewu.fusion.mapper.team.TeamMemberMapper;
import com.stonewu.fusion.mapper.team.TeamMapper;
import com.stonewu.fusion.security.SecurityUserDetails;
import com.stonewu.fusion.security.SecurityUtils;
import com.stonewu.fusion.security.TokenService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class TeamServiceTests {

    @Mock
    private TeamMapper teamMapper;

    @Mock
    private TeamMemberMapper teamMemberMapper;

    @Mock
    private TokenService tokenService;

    @InjectMocks
    private TeamService teamService;

    @AfterEach
    void clearSecurityContext() {
        SecurityContextHolder.clearContext();
    }

    @Test
    void switchCurrentTeamUpdatesTokenSessionAndSecurityContext() {
        SecurityUserDetails userDetails = new SecurityUserDetails(9L, "stone", "secret", 1, 3L, List.of());
        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken(userDetails, null, userDetails.getAuthorities()));
        when(teamMemberMapper.exists(any())).thenReturn(true);
        when(tokenService.updateCurrentTeam("token-1", 9L, 5L)).thenReturn(true);

        teamService.switchCurrentTeam(9L, 5L, "token-1");

        verify(tokenService).updateCurrentTeam("token-1", 9L, 5L);
        assertThat(SecurityUtils.getCurrentTeamId()).isEqualTo(5L);
    }

    @Test
    void switchCurrentTeamRejectsNonMemberTeam() {
        when(teamMemberMapper.exists(any())).thenReturn(false);

        assertThatThrownBy(() -> teamService.switchCurrentTeam(9L, 5L, "token-1"))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("无权切换到该团队");
    }

    @Test
    void getRequiredCurrentOwnerScopeByUserReturnsTeamScope() {
        when(teamMemberMapper.selectOne(any())).thenReturn(TeamMember.builder().teamId(5L).userId(9L).status(1).build());

        TeamService.OwnerScope ownerScope = teamService.getRequiredCurrentOwnerScopeByUser(9L);

        assertThat(ownerScope.getOwnerType()).isEqualTo(2);
        assertThat(ownerScope.getOwnerId()).isEqualTo(5L);
    }
}