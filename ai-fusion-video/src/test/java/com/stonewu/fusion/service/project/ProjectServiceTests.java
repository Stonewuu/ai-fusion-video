package com.stonewu.fusion.service.project;

import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.mapper.project.ProjectMemberMapper;
import com.stonewu.fusion.service.team.TeamService;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ProjectServiceTests {
    @Mock private ProjectMapper projectMapper;
    @Mock private ProjectMemberMapper memberMapper;
    @Mock private TeamService teamService;
    @InjectMocks private ProjectService projectService;

    @BeforeAll
    static void initializeMybatisTableMetadata() {
        TableInfoHelper.initTableInfo(new MapperBuilderAssistant(new MybatisConfiguration(), ""), Project.class);
    }

    // 创建、初始化、删除、媒体地址与标题同步已迁至 ResourceCommandIT 的真实数据库事务用例。
    @Test
    void listAccessibleByUserUsesCurrentTeamScope() {
        when(teamService.getCurrentTeamIdByUser(9L)).thenReturn(5L);
        when(teamService.listMemberUserIds(5L)).thenReturn(List.of(9L, 10L));
        when(projectMapper.selectList(any())).thenReturn(List.of());
        projectService.listAccessibleByUser(9L);
        ArgumentCaptor<LambdaQueryWrapper<Project>> wrapperCaptor = ArgumentCaptor.forClass(LambdaQueryWrapper.class);
        verify(projectMapper).selectList(wrapperCaptor.capture());
        verify(teamService).getCurrentTeamIdByUser(9L);
        verify(teamService).listMemberUserIds(5L);
    }

    @Test
    void canAccessProjectAllowsCurrentTeamPersonalProject() {
        when(projectMapper.selectById(11L)).thenReturn(Project.builder().id(11L).ownerType(1).ownerId(10L).build());
        when(memberMapper.exists(any())).thenReturn(false);
        when(teamService.getCurrentTeamIdByUser(9L)).thenReturn(5L);
        when(teamService.listMemberUserIds(5L)).thenReturn(List.of(9L, 10L));
        assertThat(projectService.canAccessProject(11L, 9L)).isTrue();
    }

    // 成员修改及删除竞态由 ProjectMembershipIT 覆盖真实事务与权限边界。
}
