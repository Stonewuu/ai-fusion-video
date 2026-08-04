package com.stonewu.fusion.service.ai.tool.project;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.stonewu.fusion.entity.project.Project;
import com.stonewu.fusion.service.ai.ToolExecutionContext;
import com.stonewu.fusion.service.project.ProjectService;
import com.stonewu.fusion.service.system.SystemConfigService;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ProjectQueryToolExecutorTests {

    private final ProjectService projectService = mock(ProjectService.class);
    private final SystemConfigService systemConfigService = mock(SystemConfigService.class);
    private final ProjectQueryToolExecutor executor =
            new ProjectQueryToolExecutor(projectService, systemConfigService);

    @Test
    void keepsLocalArtStyleResourceForDataUriTransportWhenPublicUrlIsUnavailable() {
        Project project = Project.builder()
                .id(11L)
                .name("Data URI project")
                .ownerType(1)
                .ownerId(7L)
                .artStyle("custom")
                .artStyleImageUrl("/media/art-style/reference.png")
                .build();
        when(projectService.getById(11L)).thenReturn(project);
        when(projectService.canAccessProject(project, 7L)).thenReturn(true);
        when(systemConfigService.resolvePublicUrl("/media/art-style/reference.png")).thenReturn(null);

        JSONObject result = JSONUtil.parseObj(executor.execute(
                "{\"projectId\":11}",
                ToolExecutionContext.builder().userId(7L).build()));

        JSONObject artStyleInfo = result.getJSONObject("artStyleInfo");
        assertThat(artStyleInfo.getStr("referenceImageUrl"))
                .isEqualTo("/media/art-style/reference.png");
        assertThat(artStyleInfo.getBool("referenceImageAvailable")).isTrue();
        assertThat(artStyleInfo.containsKey("referenceImageWarning")).isFalse();
    }

    @Test
    void prefersResolvedPublicArtStyleUrlWhenAvailable() {
        Project project = Project.builder()
                .id(12L)
                .name("Public URL project")
                .ownerType(1)
                .ownerId(7L)
                .artStyle("custom")
                .artStyleImageUrl("/media/art-style/reference.png")
                .build();
        when(projectService.getById(12L)).thenReturn(project);
        when(projectService.canAccessProject(project, 7L)).thenReturn(true);
        when(systemConfigService.resolvePublicUrl("/media/art-style/reference.png"))
                .thenReturn("https://assets.example.com/art-style/reference.png");

        JSONObject result = JSONUtil.parseObj(executor.execute(
                "{\"projectId\":12}",
                ToolExecutionContext.builder().userId(7L).build()));

        JSONObject artStyleInfo = result.getJSONObject("artStyleInfo");
        assertThat(artStyleInfo.getStr("referenceImageUrl"))
                .isEqualTo("https://assets.example.com/art-style/reference.png");
        assertThat(artStyleInfo.getBool("referenceImageAvailable")).isTrue();
    }
}
