package com.stonewu.fusion.service.project;

import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.core.metadata.TableInfoHelper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.mapper.asset.AssetItemMapper;
import com.stonewu.fusion.mapper.asset.AssetMapper;
import com.stonewu.fusion.mapper.project.ProjectMapper;
import com.stonewu.fusion.mapper.project.ProjectMemberMapper;
import com.stonewu.fusion.mapper.script.ScriptEpisodeMapper;
import com.stonewu.fusion.mapper.script.ScriptMapper;
import com.stonewu.fusion.mapper.script.ScriptSceneItemMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardEpisodeMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardItemMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardMapper;
import com.stonewu.fusion.mapper.storyboard.StoryboardSceneMapper;
import com.stonewu.fusion.service.project.dto.ProjectWorkspaceOverview;
import com.stonewu.fusion.service.team.TeamService;
import org.apache.ibatis.builder.MapperBuilderAssistant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.isNull;

@ExtendWith(MockitoExtension.class)
class ProjectServiceTests {

    @BeforeAll
    static void initializeMybatisTableMetadata() {
        MapperBuilderAssistant assistant = new MapperBuilderAssistant(new MybatisConfiguration(), "");
        TableInfoHelper.initTableInfo(assistant, Project.class);
        TableInfoHelper.initTableInfo(assistant, Script.class);
        TableInfoHelper.initTableInfo(assistant, Storyboard.class);
    }

    @Mock
    private ProjectMapper projectMapper;

    @Mock
    private ProjectMemberMapper memberMapper;

    @Mock
    private AssetMapper assetMapper;

    @Mock
    private AssetItemMapper assetItemMapper;

    @Mock
    private ScriptMapper scriptMapper;

    @Mock
    private ScriptEpisodeMapper scriptEpisodeMapper;

    @Mock
    private ScriptSceneItemMapper scriptSceneItemMapper;

    @Mock
    private StoryboardMapper storyboardMapper;

    @Mock
    private StoryboardEpisodeMapper storyboardEpisodeMapper;

    @Mock
    private StoryboardSceneMapper storyboardSceneMapper;

    @Mock
    private StoryboardItemMapper storyboardItemMapper;

    @Mock
    private TeamService teamService;

    @InjectMocks
    private ProjectService projectService;

    @Test
    void initializeWorkspaceCreatesOnlyMissingContainers() {
        Project project = Project.builder()
                .id(101L)
                .name("旧版本项目")
                .scope(2)
                .ownerType(2)
                .ownerId(88L)
                .build();
        when(projectMapper.selectOne(any())).thenReturn(project);
        when(scriptMapper.selectOne(any())).thenReturn(null);
        when(storyboardMapper.selectOne(any())).thenReturn(null);
        doAnswer(invocation -> {
            Script inserted = invocation.getArgument(0);
            inserted.setId(201L);
            return 1;
        }).when(scriptMapper).insert(any(Script.class));
        doAnswer(invocation -> {
            Storyboard inserted = invocation.getArgument(0);
            inserted.setId(301L);
            return 1;
        }).when(storyboardMapper).insert(any(Storyboard.class));

        ProjectWorkspaceOverview overview = projectService.initializeWorkspace(101L);

        assertThat(overview.script().getId()).isEqualTo(201L);
        assertThat(overview.script().getTitle()).isEqualTo(project.getName());
        assertThat(overview.storyboard().getId()).isEqualTo(301L);
        assertThat(overview.storyboard().getScriptId()).isEqualTo(201L);
    }

    @Test
    void initializeWorkspaceIsIdempotentWhenContainersAlreadyExist() {
        Project project = Project.builder().id(101L).name("已有工作区").build();
        Script script = Script.builder().id(201L).projectId(101L).build();
        Storyboard storyboard = Storyboard.builder()
                .id(301L)
                .projectId(101L)
                .scriptId(201L)
                .build();
        when(projectMapper.selectOne(any())).thenReturn(project);
        when(scriptMapper.selectOne(any())).thenReturn(script);
        when(storyboardMapper.selectOne(any())).thenReturn(storyboard);

        ProjectWorkspaceOverview overview = projectService.initializeWorkspace(101L);

        assertThat(overview.script()).isSameAs(script);
        assertThat(overview.storyboard()).isSameAs(storyboard);
        verify(scriptMapper, never()).insert(any(Script.class));
        verify(storyboardMapper, never()).insert(any(Storyboard.class));
    }

    @Test
    void createInitializesOneFixedScriptAndStoryboardWithProjectIdentity() {
        Project project = Project.builder()
                .name("长安十二时辰")
                .scope(2)
                .ownerType(2)
                .ownerId(88L)
                .build();
        doAnswer(invocation -> {
            Project inserted = invocation.getArgument(0);
            inserted.setId(101L);
            return 1;
        }).when(projectMapper).insert(any(Project.class));
        doAnswer(invocation -> {
            Script inserted = invocation.getArgument(0);
            inserted.setId(201L);
            return 1;
        }).when(scriptMapper).insert(any(Script.class));

        projectService.create(project);

        ArgumentCaptor<Script> scriptCaptor = ArgumentCaptor.forClass(Script.class);
        ArgumentCaptor<Storyboard> storyboardCaptor = ArgumentCaptor.forClass(Storyboard.class);
        verify(scriptMapper).insert(scriptCaptor.capture());
        verify(storyboardMapper).insert(storyboardCaptor.capture());

        Script script = scriptCaptor.getValue();
        assertThat(script.getProjectId()).isEqualTo(101L);
        assertThat(script.getTitle()).isEqualTo(project.getName());
        assertThat(script.getScope()).isEqualTo(project.getScope());
        assertThat(script.getOwnerType()).isEqualTo(project.getOwnerType());
        assertThat(script.getOwnerId()).isEqualTo(project.getOwnerId());

        Storyboard storyboard = storyboardCaptor.getValue();
        assertThat(storyboard.getProjectId()).isEqualTo(101L);
        assertThat(storyboard.getScriptId()).isEqualTo(201L);
        assertThat(storyboard.getTitle()).isEqualTo(project.getName());
        assertThat(storyboard.getScope()).isEqualTo(project.getScope());
        assertThat(storyboard.getOwnerType()).isEqualTo(project.getOwnerType());
        assertThat(storyboard.getOwnerId()).isEqualTo(project.getOwnerId());
    }

    @Test
    void createRejectsBase64ProjectImages() {
        Project project = Project.builder()
                .name("测试项目")
                .coverUrl(dataUrl("image/png"))
                .artStyleImageUrl(dataUrl("image/jpeg"))
                .build();

        assertThatThrownBy(() -> projectService.create(project))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("coverUrl 不支持 base64");
    }

    @Test
    void updateKeepsNormalImageUrlsUntouched() {
        when(projectMapper.selectById(5L)).thenReturn(Project.builder().id(5L).build());

        Project project = Project.builder()
                .id(5L)
                .coverUrl("https://example.com/project-cover.png")
                .artStyleImageUrl("/media/images/style.png")
                .build();

        projectService.update(project);

        ArgumentCaptor<Project> projectCaptor = ArgumentCaptor.forClass(Project.class);
        verify(projectMapper).updateById(projectCaptor.capture());
        assertThat(projectCaptor.getValue().getCoverUrl()).isEqualTo("https://example.com/project-cover.png");
        assertThat(projectCaptor.getValue().getArtStyleImageUrl()).isEqualTo("/media/images/style.png");
    }

    @Test
    void updateProjectNameSynchronizesFixedScriptAndStoryboardTitles() {
        when(projectMapper.selectById(5L)).thenReturn(Project.builder().id(5L).name("新项目名").build());
        Project update = Project.builder().id(5L).name("新项目名").build();

        projectService.update(update);

        ArgumentCaptor<LambdaUpdateWrapper<Script>> scriptUpdateCaptor = ArgumentCaptor.forClass(
                LambdaUpdateWrapper.class);
        ArgumentCaptor<LambdaUpdateWrapper<Storyboard>> storyboardUpdateCaptor = ArgumentCaptor.forClass(
                LambdaUpdateWrapper.class);
        verify(scriptMapper).update(isNull(), scriptUpdateCaptor.capture());
        verify(storyboardMapper).update(isNull(), storyboardUpdateCaptor.capture());
        assertThat(scriptUpdateCaptor.getValue().getParamNameValuePairs()).containsValue("新项目名");
        assertThat(storyboardUpdateCaptor.getValue().getParamNameValuePairs()).containsValue("新项目名");
    }

    @Test
    void listAccessibleByUserUsesCurrentTeamScope() {
        when(teamService.getCurrentTeamIdByUser(9L)).thenReturn(5L);
        when(teamService.listMemberUserIds(5L)).thenReturn(java.util.List.of(9L, 10L));
        when(projectMapper.selectList(any())).thenReturn(java.util.List.of());

        projectService.listAccessibleByUser(9L);

        ArgumentCaptor<com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper<Project>> wrapperCaptor =
                ArgumentCaptor.forClass(com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper.class);
        verify(projectMapper).selectList(wrapperCaptor.capture());
        verify(teamService).getCurrentTeamIdByUser(9L);
        verify(teamService).listMemberUserIds(5L);
    }

    @Test
    void canAccessProjectAllowsCurrentTeamPersonalProject() {
        when(projectMapper.selectById(11L)).thenReturn(Project.builder()
                .id(11L)
                .ownerType(1)
                .ownerId(10L)
                .build());
        when(memberMapper.exists(any())).thenReturn(false);
        when(teamService.getCurrentTeamIdByUser(9L)).thenReturn(5L);
        when(teamService.listMemberUserIds(5L)).thenReturn(java.util.List.of(9L, 10L));

        assertThat(projectService.canAccessProject(11L, 9L)).isTrue();
    }

    private static String dataUrl(String mimeType) {
        return "data:" + mimeType + ";base64,dGVzdA==";
    }
}
