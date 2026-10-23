package com.stonewu.fusion.controller.storyboard;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.common.GlobalExceptionHandler;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.StoryboardResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import com.stonewu.fusion.service.resource.model.StoryboardChildrenCommand;
import com.stonewu.fusion.service.storyboard.VideoComposeService;
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

class StoryboardCommandControllerTests {
    private ResourceCommandService commands;
    private ResourceQueryService resources;
    private VideoComposeService compose;
    private MockMvc mvc;
    private final ResourceCommandContext actor = ResourceCommandContext.user(new CanvasActor(42, 7L), null);

    @BeforeEach
    void setup() {
        commands = mock(ResourceCommandService.class);
        resources = mock(ResourceQueryService.class);
        compose = mock(VideoComposeService.class);
        CanvasActorResolver actors = mock(CanvasActorResolver.class);
        when(actors.current()).thenReturn(actor.actor());
        mvc = MockMvcBuilders.standaloneSetup(new StoryboardController(commands, resources, mock(StoryboardResourceQueryService.class), actors, compose))
                .setControllerAdvice(new StoryboardCommandExceptionHandler(), new GlobalExceptionHandler()).build();
    }

    @Test
    void assetsPatchPreservesEmptyArraysExplicitNullMissingFieldsAndObservedVersion() throws Exception {
        when(commands.patch(any(), eq(actor))).thenReturn(receipt());
        mvc.perform(patch("/api/storyboard/item/12/assets").contentType(MediaType.APPLICATION_JSON).content("""
                {"version":4,"requestId":"save-1","patch":{"sceneAssetItemId":null,"characterIds":[]}}
                """)).andExpect(status().isOk()).andExpect(jsonPath("$.data.version").value(5));
        var captor = ArgumentCaptor.forClass(ResourcePatchCommand.class);
        verify(commands).patch(captor.capture(), eq(actor));
        var value = captor.getValue();
        assertThat(value.resourceType()).isEqualTo(CanvasValueType.SHOT);
        assertThat(value.resourceId()).isEqualTo(12);
        assertThat(value.expectedVersion()).isEqualTo(4);
        assertThat(value.patch().get("sceneAssetItemId").isNull()).isTrue();
        assertThat(value.patch().get("characterIds").isArray()).isTrue();
        assertThat(value.patch().has("propIds")).isFalse();
    }

    @Test
    void frameUpdateChangesOnlySpecifiedFrameFieldsAndDoesNotClearAnOmittedPrompt() throws Exception {
        when(commands.patch(any(), eq(actor))).thenReturn(receipt());
        mvc.perform(put("/api/storyboard/item/12/updateFrame").contentType(MediaType.APPLICATION_JSON).content("""
                {"version":4,"requestId":"frame-1","patch":{"frameType":"last","imageUrl":null}}
                """)).andExpect(status().isOk());
        var captor = ArgumentCaptor.forClass(ResourcePatchCommand.class);
        verify(commands).patch(captor.capture(), eq(actor));
        assertThat(captor.getValue().patch().size()).isEqualTo(1);
        assertThat(captor.getValue().patch().get("lastFrameImageUrl").isNull()).isTrue();
        mvc.perform(put("/api/storyboard/item/12/updateFrame").contentType(MediaType.APPLICATION_JSON).content("""
                {"version":4,"requestId":"frame-2","patch":{"frameType":"middle","imageUrl":null}}
                """)).andExpect(status().isBadRequest());
        verifyNoMoreInteractions(commands);
    }

    @Test
    void unversionedAndNullVersionWritesAreRejectedBeforeAnyCommand() throws Exception {
        mvc.perform(put("/api/storyboard/item").contentType(MediaType.APPLICATION_JSON).content("{\"id\":1,\"content\":\"旧协议\"}"))
                .andExpect(status().isBadRequest());
        mvc.perform(post("/api/storyboard/episode").contentType(MediaType.APPLICATION_JSON).content("""
                {"parentId":1,"parentVersion":null,"requestId":"create-1","fields":{"episodeNumber":1}}
                """)).andExpect(status().isBadRequest());
        mvc.perform(post("/api/storyboard/episode/1/clearContent").contentType(MediaType.APPLICATION_JSON).content("{\"requestId\":\"clear-1\"}"))
                .andExpect(status().isBadRequest());
        mvc.perform(delete("/api/storyboard/item/12").param("requestId", "delete-1")).andExpect(status().isBadRequest());
        verifyNoInteractions(commands);
    }

    @Test
    void createRejectsASecondParentInsideTheFieldPatch() throws Exception {
        when(commands.execute(any(), eq(actor))).thenReturn(receipt());
        mvc.perform(post("/api/storyboard/scene").contentType(MediaType.APPLICATION_JSON).content("""
                {"parentId":3,"parentVersion":7,"requestId":"create-1","fields":{"sceneHeading":"客厅"}}
                """)).andExpect(status().isOk());
        var captor = ArgumentCaptor.forClass(ResourceCreateCommand.class);
        verify(commands).execute(captor.capture(), eq(actor));
        assertThat(captor.getValue().fields().path("episodeId").longValue()).isEqualTo(3);
        assertThat(captor.getValue().fields().path("parentVersion").intValue()).isEqualTo(7);
        mvc.perform(post("/api/storyboard/scene").contentType(MediaType.APPLICATION_JSON).content("""
                {"parentId":3,"parentVersion":7,"requestId":"create-2","fields":{"episodeId":99,"sceneHeading":"越界"}}
                """)).andExpect(status().isBadRequest());
        verifyNoMoreInteractions(commands);
    }

    @Test
    void clearUsesTheCallersCollectionVersionAndReturnsTheCommittedParent() throws Exception {
        var result = JsonNodeFactory.instance.objectNode();
        result.set("parent", JsonNodeFactory.instance.objectNode().put("id", 3).put("version", 8));
        when(commands.execute(any(), eq(actor))).thenReturn(new ResourceCommandReceipt("clear-command", "clear-1", result, List.of()));
        mvc.perform(post("/api/storyboard/episode/3/clearContent").contentType(MediaType.APPLICATION_JSON).content("""
                {"version":7,"requestId":"clear-1"}
                """)).andExpect(status().isOk()).andExpect(jsonPath("$.data.version").value(8));
        var captor = ArgumentCaptor.forClass(StoryboardChildrenCommand.class);
        verify(commands).execute(captor.capture(), eq(actor));
        assertThat(captor.getValue().mode()).isEqualTo(StoryboardChildrenCommand.Mode.CLEAR);
        assertThat(captor.getValue().expectedVersion()).isEqualTo(7);
        assertThat(captor.getValue().resourceType()).isEqualTo(CanvasValueType.STORYBOARD_EPISODE);
        verifyNoInteractions(resources);
    }

    @Test
    void composeChecksResourceAccessBeforeSubmittingAnyWork() throws Exception {
        when(resources.get(CanvasValueType.STORYBOARD_EPISODE, 3, actor.actor()))
                .thenThrow(new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "无权读取分镜", Map.of()));
        mvc.perform(post("/api/storyboard/episode/3/compose-video").contentType(MediaType.APPLICATION_JSON).content("{\"version\":0,\"requestId\":\"compose-test\",\"includeBgm\":true}")).andExpect(status().isForbidden());
        verifyNoInteractions(compose);
    }

    private ResourceCommandReceipt receipt() {
        return new ResourceCommandReceipt("committed", "save-1", JsonNodeFactory.instance.objectNode().put("id", 12).put("version", 5), List.of());
    }
}
