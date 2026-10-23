package com.stonewu.fusion.controller.canvas;

import com.stonewu.fusion.common.GlobalExceptionHandler;
import com.stonewu.fusion.security.SecurityUserDetails;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.CanvasCommandService;
import com.stonewu.fusion.service.canvas.CanvasEventStream;
import com.stonewu.fusion.service.canvas.CanvasQueryService;
import com.stonewu.fusion.service.canvas.agent.node.CanvasAgentProfiles;
import com.stonewu.fusion.service.ai.configuration.AgentSkillSelection;
import com.stonewu.fusion.service.canvas.capability.CanvasCapabilityRegistry;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.canvas.command.CanvasEditCommand;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.team.TeamService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.asyncDispatch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class CanvasControllerTests {
    private final String canvas = UUID.randomUUID().toString();
    private final CanvasActor actor = new CanvasActor(42, 7L);
    private CanvasCommandService commands;
    private CanvasQueryService queries;
    private CanvasEventStream events;
    private MockMvc mvc;

    @BeforeEach
    void setup() {
        commands = mock(CanvasCommandService.class);
        queries = mock(CanvasQueryService.class);
        events = mock(CanvasEventStream.class);
        TeamService teams = mock(TeamService.class);
        when(teams.getCurrentTeamIdByUser(42L)).thenReturn(7L);
        CanvasController controller = new CanvasController(new CanvasActorResolver(teams), commands, queries,
                new CanvasCapabilityRegistry(), events, mock(CanvasModelCatalog.class), mock(CanvasAgentProfiles.class), mock(AgentSkillSelection.class));
        mvc = MockMvcBuilders.standaloneSetup(controller)
                .setControllerAdvice(new CanvasExceptionHandler(), new GlobalExceptionHandler()).build();
        SecurityUserDetails principal = new SecurityUserDetails(42L, "creator", "secret", 1, 7L, List.of());
        SecurityContextHolder.getContext().setAuthentication(new UsernamePasswordAuthenticationToken(principal, null, principal.getAuthorities()));
    }

    @AfterEach
    void cleanup() { SecurityContextHolder.clearContext(); }

    @Test
    void editUsesAuthenticatedIdentityAndTeam() throws Exception {
        mvc.perform(post("/api/canvases/{id}/commands", canvas).contentType(MediaType.APPLICATION_JSON).content("""
                {"requestId":"rename-1","operations":[{"type":"canvas.node.patch","nodeId":"node-id","expectedConfigVersion":0,"title":"改名"}]}
                """)).andExpect(status().isOk());
        verify(commands).edit(eq(canvas), any(CanvasEditCommand.class), eq(actor));
    }

    @Test
    void missingExpectedVersionIsAnInvalidRequestInsteadOfVersionZero() throws Exception {
        mvc.perform(post("/api/canvases/{id}/commands", canvas).contentType(MediaType.APPLICATION_JSON).content("""
                {"requestId":"rename-2","operations":[{"type":"canvas.node.patch","nodeId":"node-id","title":"缺少版本"}]}
                """)).andExpect(status().isBadRequest()).andExpect(jsonPath("$.data.errorCode").value("INVALID_COMMAND"));
        verifyNoInteractions(commands);
    }

    @Test
    void explicitNullVersionIsNotCoercedToZero() throws Exception {
        mvc.perform(post("/api/canvases/{id}/commands", canvas).contentType(MediaType.APPLICATION_JSON).content("""
                {"requestId":"rename-3","operations":[{"type":"canvas.node.patch","nodeId":"node-id","expectedConfigVersion":null,"title":"空版本"}]}
                """)).andExpect(status().isBadRequest()).andExpect(jsonPath("$.data.errorCode").value("INVALID_COMMAND"));
        verifyNoInteractions(commands);
    }

    @Test
    void preservesConflictDetailsInErrorEnvelope() throws Exception {
        when(queries.snapshot(canvas, actor)).thenThrow(new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                "版本已变化", Map.of("currentVersion", 8)));
        mvc.perform(get("/api/canvases/{id}", canvas)).andExpect(status().isConflict())
                .andExpect(jsonPath("$.data.errorCode").value("VERSION_CONFLICT"))
                .andExpect(jsonPath("$.data.details.currentVersion").value(8));
    }

    @Test
    void rejectedStreamReturnsStructuredJsonBeforeOpeningSse() throws Exception {
        when(events.open(canvas, actor, 0)).thenThrow(new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "无权访问", Map.of()));
        mvc.perform(get("/api/canvases/{id}/stream", canvas).accept(MediaType.TEXT_EVENT_STREAM))
                .andExpect(status().isForbidden()).andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.data.errorCode").value("ACCESS_DENIED"));
    }

    @Test
    void resumesFromLastEventIdAndKeepsCanvasEventNamespace() throws Exception {
        when(events.open(canvas, actor, 44)).thenReturn(Flux.just(ServerSentEvent.<Object>builder(Map.of("sequence", 45))
                .id("45").event("canvas.graph.changed").build()));
        MvcResult pending = mvc.perform(get("/api/canvases/{id}/stream", canvas)
                .param("afterSequence", "1").header("Last-Event-ID", "44").accept(MediaType.TEXT_EVENT_STREAM))
                .andReturn();
        pending.getAsyncResult(5000);
        MvcResult result = mvc.perform(asyncDispatch(pending)).andExpect(status().isOk()).andReturn();
        assertThat(result.getResponse().getContentAsString()).contains("id:45", "event:canvas.graph.changed");
        verify(events).open(canvas, actor, 44);
    }

    @Test
    void rejectsMalformedReconnectCursor() throws Exception {
        mvc.perform(get("/api/canvases/{id}/stream", canvas).header("Last-Event-ID", "agent-run:1"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.data.errorCode").value("INVALID_CURSOR"));
        verifyNoInteractions(events);
    }
}
