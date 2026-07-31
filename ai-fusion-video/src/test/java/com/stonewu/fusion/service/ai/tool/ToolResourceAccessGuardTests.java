package com.stonewu.fusion.service.ai.tool;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.entity.script.Script;
import com.stonewu.fusion.entity.storyboard.Storyboard;
import com.stonewu.fusion.entity.storyboard.StoryboardEpisode;
import com.stonewu.fusion.entity.storyboard.StoryboardItem;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.script.ScriptService;
import com.stonewu.fusion.service.storyboard.StoryboardService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class ToolResourceAccessGuardTests {

    @Mock
    private ProjectService projectService;

    @Mock
    private ScriptService scriptService;

    @Mock
    private StoryboardService storyboardService;

    private ToolResourceAccessGuard accessGuard;

    @BeforeEach
    void setUp() {
        accessGuard = new ToolResourceAccessGuard(projectService, scriptService, storyboardService);
    }

    @Test
    void rejectsScriptWhenCurrentUserCannotAccessItsProject() {
        Script script = Script.builder().id(11L).projectId(21L).build();
        Project project = Project.builder().id(21L).build();
        when(scriptService.getById(11L)).thenReturn(script);
        when(projectService.getById(21L)).thenReturn(project);
        when(projectService.canAccessProject(project, 31L)).thenReturn(false);

        assertThatThrownBy(() -> accessGuard.requireScript(11L, 31L))
                .isInstanceOf(BusinessException.class)
                .hasMessage("资源不存在或无权访问");
    }

    @Test
    void resolvesStoryboardItemThroughItsProject() {
        StoryboardItem item = StoryboardItem.builder().id(41L).storyboardId(51L).build();
        Storyboard storyboard = Storyboard.builder().id(51L).projectId(61L).build();
        Project project = Project.builder().id(61L).build();
        when(storyboardService.getItemById(41L)).thenReturn(item);
        when(storyboardService.getById(51L)).thenReturn(storyboard);
        when(projectService.getById(61L)).thenReturn(project);
        when(projectService.canAccessProject(project, 31L)).thenReturn(true);

        assertThat(accessGuard.requireStoryboardItem(41L, 31L)).isSameAs(item);
    }

    @Test
    void rejectsStoryboardEpisodeFromAnotherStoryboard() {
        StoryboardEpisode episode = StoryboardEpisode.builder().id(71L).storyboardId(81L).build();
        Storyboard storyboard = Storyboard.builder().id(81L).projectId(91L).build();
        Project project = Project.builder().id(91L).build();
        when(storyboardService.getEpisodeById(71L)).thenReturn(episode);
        when(storyboardService.getById(81L)).thenReturn(storyboard);
        when(projectService.getById(91L)).thenReturn(project);
        when(projectService.canAccessProject(project, 31L)).thenReturn(true);

        assertThatThrownBy(() -> accessGuard.requireStoryboardEpisode(82L, 71L, 31L))
                .isInstanceOf(BusinessException.class)
                .hasMessage("分镜集不属于指定分镜");
    }
}
