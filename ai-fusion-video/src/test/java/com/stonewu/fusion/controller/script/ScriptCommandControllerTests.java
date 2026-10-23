package com.stonewu.fusion.controller.script;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.common.GlobalExceptionHandler;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import java.util.List;
import java.util.Map;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

class ScriptCommandControllerTests {
    private ResourceCommandService commands;
    private MockMvc mvc;
    private final ResourceCommandContext actor = ResourceCommandContext.user(new CanvasActor(42, 7L), null);

    @BeforeEach
    void setup() {
        commands = mock(ResourceCommandService.class);
        CanvasActorResolver actors = mock(CanvasActorResolver.class);
        when(actors.current()).thenReturn(actor.actor());
        mvc = MockMvcBuilders.standaloneSetup(new ScriptController(commands, mock(ResourceQueryService.class), actors))
                .setControllerAdvice(new ScriptCommandExceptionHandler(), new GlobalExceptionHandler()).build();
    }

    @Test
    void scenePatchPreservesVersionStructuredDialoguesNullAndAuthenticatedIdentity() throws Exception {
        when(commands.patch(any(), eq(actor))).thenReturn(new ResourceCommandReceipt("committed", "save-1",
                JsonNodeFactory.instance.objectNode().put("id", 12).put("version", 5), List.of()));
        mvc.perform(put("/api/script/scene").contentType(MediaType.APPLICATION_JSON).content("""
                {"id":12,"version":4,"requestId":"save-1","patch":{"location":null,"dialogues":[{"type":1,"character_name":"甲","content":"你好"}]}}
                """)).andExpect(status().isOk()).andExpect(jsonPath("$.data.version").value(5));
        var captor = ArgumentCaptor.forClass(ResourcePatchCommand.class);
        verify(commands).patch(captor.capture(), eq(actor));
        assertThat(captor.getValue().resourceType()).isEqualTo(CanvasValueType.SCRIPT_SCENE);
        assertThat(captor.getValue().expectedVersion()).isEqualTo(4);
        assertThat(captor.getValue().patch().get("location").isNull()).isTrue();
        assertThat(captor.getValue().patch().get("dialogues").isArray()).isTrue();
    }

    @Test
    void oldUpdateAndMissingDeleteVersionAndNullVersionAreExplicitClientErrors() throws Exception {
        mvc.perform(put("/api/script").contentType(MediaType.APPLICATION_JSON).content("{\"id\":1,\"rawContent\":\"旧协议\"}"))
                .andExpect(status().isBadRequest());
        mvc.perform(delete("/api/script/scene/12").param("requestId", "delete-1"))
                .andExpect(status().isBadRequest()).andExpect(jsonPath("$.data.errorCode").value("INVALID_COMMAND"));
        mvc.perform(put("/api/script/episode").contentType(MediaType.APPLICATION_JSON).content("""
                {"id":2,"requestId":"save-2","version":null,"patch":{"title":"无版本"}}
                """)).andExpect(status().isBadRequest());
        verifyNoInteractions(commands);
    }

    @Test
    void createDerivesParentBindingAndRefusesHiddenParentOverrides() throws Exception {
        when(commands.execute(any(), eq(actor))).thenReturn(new ResourceCommandReceipt("created", "create-1",
                JsonNodeFactory.instance.objectNode().put("id", 20).put("version", 0), List.of()));
        mvc.perform(post("/api/script/episode").contentType(MediaType.APPLICATION_JSON).content("""
                {"parentId":3,"parentVersion":7,"requestId":"create-1","fields":{"episodeNumber":2,"title":"第二集"}}
                """)).andExpect(status().isOk());
        var captor = ArgumentCaptor.forClass(ResourceCreateCommand.class);
        verify(commands).execute(captor.capture(), eq(actor));
        assertThat(captor.getValue().fields().path("scriptId").longValue()).isEqualTo(3);
        assertThat(captor.getValue().fields().path("parentVersion").intValue()).isEqualTo(7);
        mvc.perform(post("/api/script/episode").contentType(MediaType.APPLICATION_JSON).content("""
                {"parentId":3,"parentVersion":7,"requestId":"create-2","fields":{"scriptId":8,"episodeNumber":3}}
                """)).andExpect(status().isBadRequest());
        verifyNoMoreInteractions(commands);
    }

    @Test
    void sourceReplacementReturnsConflictDetailsInsteadOfOverwriting() throws Exception {
        when(commands.execute(any(), eq(actor))).thenThrow(new CanvasException(409, CanvasErrorCode.VERSION_CONFLICT,
                "剧本已有结构修改", Map.of("currentVersion", 9)));
        mvc.perform(put("/api/script/3/source").contentType(MediaType.APPLICATION_JSON).content("""
                {"requestId":"replace-1","version":7,"rawContent":"新故事"}
                """)).andExpect(status().isConflict()).andExpect(jsonPath("$.data.details.currentVersion").value(9));
    }
}
