package com.stonewu.fusion.controller.asset;

import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.stonewu.fusion.common.GlobalExceptionHandler;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasErrorCode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.resource.AssetResourceQueryService;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.ResourceQueryService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCommandReceipt;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
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

class AssetCommandControllerTests {
    private ResourceCommandService commands;
    private ResourceQueryService resources;
    private AssetResourceQueryService assets;
    private MockMvc mvc;
    private final ResourceCommandContext actor = ResourceCommandContext.user(new CanvasActor(42, 7L), null);

    @BeforeEach
    void setup() {
        commands = mock(ResourceCommandService.class);
        resources = mock(ResourceQueryService.class);
        assets = mock(AssetResourceQueryService.class);
        CanvasActorResolver actors = mock(CanvasActorResolver.class);
        when(actors.current()).thenReturn(actor.actor());
        mvc = MockMvcBuilders.standaloneSetup(new AssetController(commands, resources, assets, actors))
                .setControllerAdvice(new AssetCommandExceptionHandler(), new GlobalExceptionHandler()).build();
    }

    @Test
    void itemPatchKeepsParentAndChildVersionNativePropertiesExplicitNullAndTrustedActor() throws Exception {
        when(commands.patch(any(), eq(actor))).thenReturn(new ResourceCommandReceipt("committed", "save-1",
                JsonNodeFactory.instance.objectNode().put("id", 12).put("version", 5).put("assetVersion", 9), List.of()));
        mvc.perform(put("/api/asset/item").contentType(MediaType.APPLICATION_JSON).content("""
                {"id":12,"version":4,"assetVersion":8,"requestId":"save-1","patch":{"imageUrl":null,"properties":{"appearance":"夜晚"}}}
                """)).andExpect(status().isOk()).andExpect(jsonPath("$.data.version").value(5))
                .andExpect(jsonPath("$.data.assetVersion").value(9));
        var captor = ArgumentCaptor.forClass(ResourcePatchCommand.class);
        verify(commands).patch(captor.capture(), eq(actor));
        assertThat(captor.getValue().resourceType()).isEqualTo(CanvasValueType.ASSET_ITEM);
        assertThat(captor.getValue().expectedVersion()).isEqualTo(4);
        assertThat(captor.getValue().expectedParentVersion()).isEqualTo(8);
        assertThat(captor.getValue().patch().get("imageUrl").isNull()).isTrue();
        assertThat(captor.getValue().patch().get("properties").isObject()).isTrue();
    }

    @Test
    void oldUpdatesNullVersionsMissingParentAndImplicitReuseAreClientErrors() throws Exception {
        mvc.perform(put("/api/asset").contentType(MediaType.APPLICATION_JSON).content("{\"id\":1,\"name\":\"旧协议\"}"))
                .andExpect(status().isBadRequest());
        mvc.perform(put("/api/asset/item").contentType(MediaType.APPLICATION_JSON).content("""
                {"id":1,"version":0,"requestId":"x","patch":{"name":"缺父版本"}}
                """)).andExpect(status().isBadRequest());
        mvc.perform(put("/api/asset/item").contentType(MediaType.APPLICATION_JSON).content("""
                {"id":1,"version":null,"assetVersion":0,"requestId":"x","patch":{"name":"无版本"}}
                """)).andExpect(status().isBadRequest());
        mvc.perform(delete("/api/asset/item/12").param("requestId", "delete-1").param("version", "2"))
                .andExpect(status().isBadRequest());
        mvc.perform(post("/api/asset/batch").contentType(MediaType.APPLICATION_JSON).content("""
                {"resourceType":"asset","requestId":"batch-1","entries":[{"type":"prop","name":"重复策略缺失"}]}
                """)).andExpect(status().isBadRequest());
        verifyNoInteractions(commands);
    }

    @Test
    void childCreateCannotOverrideTheExplicitParentAndVersion() throws Exception {
        when(commands.execute(any(), eq(actor))).thenReturn(new ResourceCommandReceipt("created", "create-1",
                JsonNodeFactory.instance.objectNode().put("id", 20).put("version", 0).put("assetVersion", 8), List.of()));
        mvc.perform(post("/api/asset/item").contentType(MediaType.APPLICATION_JSON).content("""
                {"assetId":3,"parentVersion":7,"requestId":"create-1","fields":{"name":"变体","itemType":"variant"}}
                """)).andExpect(status().isOk());
        var captor = ArgumentCaptor.forClass(ResourceCreateCommand.class);
        verify(commands).execute(captor.capture(), eq(actor));
        assertThat(captor.getValue().fields().path("assetId").longValue()).isEqualTo(3);
        assertThat(captor.getValue().fields().path("parentVersion").intValue()).isEqualTo(7);
        mvc.perform(post("/api/asset/item").contentType(MediaType.APPLICATION_JSON).content("""
                {"assetId":3,"parentVersion":7,"requestId":"create-2","fields":{"assetId":8,"name":"偷偷换父资产"}}
                """)).andExpect(status().isBadRequest());
        verifyNoMoreInteractions(commands);
    }

    @Test
    void globalListUsesTrustedScopeAndDetailDenialIsNotAnEmptySuccess() throws Exception {
        when(assets.list(null, null, null, true, actor.actor())).thenReturn(List.of());
        mvc.perform(get("/api/asset/list-with-items")).andExpect(status().isOk()).andExpect(jsonPath("$.data").isArray());
        verify(assets).list(null, null, null, true, actor.actor());
        when(resources.get(CanvasValueType.ASSET, 3, actor.actor()))
                .thenThrow(new CanvasException(403, CanvasErrorCode.ACCESS_DENIED, "无权读取素材", Map.of()));
        mvc.perform(get("/api/asset/3")).andExpect(status().isForbidden())
                .andExpect(jsonPath("$.data.errorCode").value("ACCESS_DENIED"));
    }

    @Test
    void declaredExtensionTypesHaveAnExplicitEmptySchemaButUnknownTypesAreRejected() throws Exception {
        mvc.perform(get("/api/asset/metadata/vehicle")).andExpect(status().isOk()).andExpect(jsonPath("$.data.fields").isEmpty());
        mvc.perform(get("/api/asset/metadata/character")).andExpect(status().isOk()).andExpect(jsonPath("$.data.fields").isNotEmpty());
        mvc.perform(get("/api/asset/metadata/unknown")).andExpect(status().isBadRequest());
    }
}
