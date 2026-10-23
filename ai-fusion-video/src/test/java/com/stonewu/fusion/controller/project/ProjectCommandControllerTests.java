package com.stonewu.fusion.controller.project;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.common.GlobalExceptionHandler;
import com.stonewu.fusion.controller.canvas.CanvasExceptionHandler;
import com.stonewu.fusion.controller.canvas.CanvasResourceController;
import com.stonewu.fusion.security.SecurityUserDetails;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.CanvasPlacementService;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceAccessPolicy;
import com.stonewu.fusion.repository.canvas.CanvasDocumentRepository;
import com.stonewu.fusion.service.canvas.CanvasAccessPolicy;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import com.stonewu.fusion.service.system.SystemConfigService;
import com.stonewu.fusion.service.team.TeamService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.MediaType;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class ProjectCommandControllerTests {
    private ResourceCommandService commands;
    private MockMvc mvc;

    @BeforeEach
    void setup() {
        commands = mock(ResourceCommandService.class);
        TeamService teams = mock(TeamService.class);
        when(teams.getCurrentTeamIdByUser(42L)).thenReturn(7L);
        CanvasActorResolver actors = new CanvasActorResolver(teams);
        mvc = MockMvcBuilders.standaloneSetup(
                        new ProjectController(mock(SystemConfigService.class), commands, actors, mock(ResourceQueryService.class)),
                        new CanvasResourceController(commands, actors, mock(ResourceQueryService.class),
                                mock(CanvasDocumentRepository.class), mock(CanvasAccessPolicy.class), mock(CanvasPlacementService.class), mock(ResourceAccessPolicy.class)))
                .setControllerAdvice(new ProjectCommandExceptionHandler(), new CanvasExceptionHandler(), new GlobalExceptionHandler()).build();
        SecurityUserDetails principal = new SecurityUserDetails(42L, "user", "test", 1, 7L, List.of());
        SecurityContextHolder.getContext().setAuthentication(new UsernamePasswordAuthenticationToken(principal, null, principal.getAuthorities()));
    }

    @AfterEach
    void cleanup() { SecurityContextHolder.clearContext(); }

    @Test
    void settingsUpdateUsesAuthenticatedCommandContextAndPreservesExplicitNull() throws Exception {
        when(commands.patch(any(), any())).thenReturn(new ResourceCommandReceipt("command", "save-1",
                JsonNodeFactory.instance.objectNode().put("id", 8).put("version", 3), List.of()));
        mvc.perform(put("/api/project").contentType(MediaType.APPLICATION_JSON).content("""
                {"id":8,"version":2,"requestId":"save-1","patch":{"name":"新名称","artStyle":null,"properties":{"aspectRatio":"16:9"}}}
                """)).andExpect(status().isOk()).andExpect(jsonPath("$.data.version").value(3));
        ArgumentCaptor<ResourcePatchCommand> command = ArgumentCaptor.forClass(ResourcePatchCommand.class);
        ArgumentCaptor<ResourceCommandContext> context = ArgumentCaptor.forClass(ResourceCommandContext.class);
        verify(commands).patch(command.capture(), context.capture());
        assertThat(command.getValue().resourceType()).isEqualTo(CanvasValueType.PROJECT);
        assertThat(command.getValue().expectedVersion()).isEqualTo(2);
        assertThat(command.getValue().patch().path("artStyle").isNull()).isTrue();
        assertThat(context.getValue()).isEqualTo(ResourceCommandContext.user(new CanvasActor(42, 7L), null));
    }

    @Test
    void oldUnversionedUpdateIsRejected() throws Exception {
        mvc.perform(put("/api/project").contentType(MediaType.APPLICATION_JSON).content("{\"id\":8,\"name\":\"无版本写入\"}"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.data.errorCode").value("INVALID_COMMAND"));
        verifyNoInteractions(commands);
    }

    @Test
    void canvasUsesSameCommandsAndReturnsConflictDetails() throws Exception {
        when(commands.patch(any(), any())).thenThrow(new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                "项目已变化", Map.of("currentVersion", 3)));
        String id = "ab6468d1-6eae-4d2b-8565-8d985d8d9824";
        mvc.perform(post("/api/canvases/{id}/resource-commands", id).contentType(MediaType.APPLICATION_JSON).content("""
                {"requestId":"canvas-1","resourceType":"project","resourceId":8,"expectedVersion":2,"patch":{"name":"新名称"}}
                """)).andExpect(status().isConflict()).andExpect(jsonPath("$.data.details.currentVersion").value(3));
        ArgumentCaptor<ResourceCommandContext> context = ArgumentCaptor.forClass(ResourceCommandContext.class);
        verify(commands).patch(any(), context.capture());
        assertThat(context.getValue().canvasId()).isEqualTo(id);
    }
}
