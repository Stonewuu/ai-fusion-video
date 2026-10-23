package com.stonewu.fusion.integration;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.service.canvas.model.CanvasException;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.MediaContentReader;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaSourceImportService;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaSourceReference;
import com.stonewu.fusion.service.resource.ResourceCommandService;
import com.stonewu.fusion.service.resource.model.ResourceBindMediaCommand;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import com.stonewu.fusion.service.resource.model.ResourceCreateCommand;
import com.stonewu.fusion.service.resource.model.ResourcePatchCommand;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.nio.file.Files;
import java.util.Map;
import java.util.UUID;
import static com.stonewu.fusion.service.canvas.model.CanvasValueType.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@Testcontainers(disabledWithoutDocker = false)
class MediaBusinessBindingIT extends MediaResourceTestContext {
    @Test
    void bindingSharesOnlyThroughLiveBusinessPermissionAndUnbindingRevokesTicket() throws Exception {
        var media = upload(UUID.randomUUID().toString(), png, "私有源.png", null, USER);
        long asset = asset(71L, USER);
        assertThatThrownBy(() -> queries.get(media.id(), OTHER)).isInstanceOf(MediaException.class);
        var command = binding(ASSET, asset, 0, null, Map.of("coverUrl", media.id()));
        var saved = commands().execute(command, USER);
        assertThat(commands().execute(command, USER)).isEqualTo(saved);
        assertThat(saved.result().path("coverUrl").textValue()).isEqualTo(MediaReference.of(media.id()));
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_binding WHERE media_id = ?", Integer.class, media.id())).isEqualTo(1);
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (71, 72, 3)");
        assertThat(queries.get(media.id(), OTHER).ownerId()).isEqualTo(71);
        var ticket = tickets.browser(media.id(), OTHER, false);
        String token = ticket.url().substring(ticket.url().indexOf("?ticket=") + 8);
        assertThat(tickets.authorize(media.id(), token).actor()).isEqualTo(OTHER);
        jdbc.update("DELETE FROM afv_project_member WHERE project_id = 71 AND user_id = 72");
        assertThatThrownBy(() -> tickets.authorize(media.id(), token)).isInstanceOf(MediaException.class);
        jdbc.update("INSERT INTO afv_project_member(project_id, user_id, role) VALUES (71, 72, 3)");
        commands().patch(new ResourcePatchCommand(UUID.randomUUID().toString(), ASSET, asset, 1,
                mapper.createObjectNode().putNull("coverUrl")), USER);
        assertThatThrownBy(() -> tickets.authorize(media.id(), token)).isInstanceOf(MediaException.class);
        assertThat(queries.get(media.id(), OWNER).state()).isEqualTo("READY");
    }

    @Test
    void wrongTypeStaleVersionAndGuessedPrivateResourceRollbackBusinessAndAudit() throws Exception {
        var mine = upload(UUID.randomUUID().toString(), png, "图片.png", null, USER);
        var otherContext = ResourceCommandContext.user(OTHER, null);
        var privateMedia = upload(UUID.randomUUID().toString(), png, "他人.png", null, otherContext);
        long asset = asset(null, USER);
        int auditCount = jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command", Integer.class);
        assertThatThrownBy(() -> commands().execute(binding(ASSET, asset, 0, null,
                Map.of("properties.videoUrl", mine.id())), USER)).isInstanceOf(MediaException.class);
        assertThatThrownBy(() -> commands().execute(binding(ASSET, asset, 0, null,
                Map.of("coverUrl", privateMedia.id())), USER)).isInstanceOf(MediaException.class);
        assertThatThrownBy(() -> commands().patch(new ResourcePatchCommand(UUID.randomUUID().toString(), ASSET, asset, 0,
                mapper.createObjectNode().put("coverUrl", MediaReference.of(privateMedia.id()))), USER)).isInstanceOf(MediaException.class);
        assertThat(jdbc.queryForObject("SELECT version FROM afv_asset WHERE id = ?", Integer.class, asset)).isZero();
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_resource_command", Integer.class)).isEqualTo(auditCount);
        commands().execute(binding(ASSET, asset, 0, null, Map.of("coverUrl", mine.id())), USER);
        assertThatThrownBy(() -> commands().execute(binding(ASSET, asset, 0, null, Map.of("coverUrl", mine.id())), USER))
                .isInstanceOf(CanvasException.class);
    }

    @Test
    void creationChecksNestedMediaAndAliasImportDoesNotCopyOrChangeOwnership() throws Exception {
        var media = upload(UUID.randomUUID().toString(), png, "一次接收.png", null, USER);
        var fields = mapper.createObjectNode().put("name", "带媒体素材").put("type", "character");
        fields.putObject("initialItem").put("imageUrl", MediaReference.of(media.id()));
        var saved = commands().execute(new ResourceCreateCommand(UUID.randomUUID().toString(), ASSET, 71L, fields), USER);
        long asset = saved.result().path("id").longValue();
        int version = jdbc.queryForObject("SELECT version FROM afv_asset WHERE id = ?", Integer.class, asset);
        long count = jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_blob", Long.class);
        var ref = new MediaSourceReference(ASSET, asset, version, "coverUrl");
        var imports = context.getBean(MediaSourceImportService.class);
        assertThat(imports.importSource(ref, USER).id()).isEqualTo(media.id());
        assertThat(imports.importSource(ref, USER).id()).isEqualTo(media.id());
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_media_blob", Long.class)).isEqualTo(count);
        assertThat(queries.sources(media.id(), OWNER)).contains(ref);
        var invalid = fields.deepCopy();
        ((ObjectNode) invalid.get("initialItem")).put("imageUrl", MediaReference.of(UUID.randomUUID().toString()));
        long before = jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset", Long.class);
        assertThatThrownBy(() -> commands().execute(new ResourceCreateCommand(UUID.randomUUID().toString(), ASSET, null, invalid), USER))
                .isInstanceOf(MediaException.class);
        assertThat(jdbc.queryForObject("SELECT COUNT(*) FROM afv_asset", Long.class)).isEqualTo(before);
    }

    @Test
    void parentVersionAndPropertiesArePreservedAndContentReaderUsesSameAccess() throws Exception {
        var media = upload(UUID.randomUUID().toString(), png, "读取.png", null, USER);
        long asset = asset(null, USER);
        var child = commands().execute(new ResourceCreateCommand(UUID.randomUUID().toString(), ASSET_ITEM, null,
                mapper.createObjectNode().put("assetId", asset).put("parentVersion", 0).put("name", "变体")
                        .set("properties", mapper.createObjectNode().put("customNote", "保留"))), USER).result();
        long item = child.path("id").longValue();
        assertThatThrownBy(() -> commands().execute(binding(ASSET_ITEM, item, 0, 0, Map.of("imageUrl", media.id())), USER))
                .isInstanceOf(CanvasException.class);
        var saved = commands().execute(binding(ASSET_ITEM, item, 0, 1, Map.of("imageUrl", media.id(),
                "properties.thumbnailUrl", media.id())), USER);
        assertThat(saved.result().path("properties").path("customNote").textValue()).isEqualTo("保留");
        assertThat(saved.result().path("assetVersion").intValue()).isEqualTo(2);
        var reader = context.getBean(MediaContentReader.class);
        var destination = Files.createTempFile(directory, "read-", ".png");
        try {
            reader.copy(media.id(), OWNER, destination, 1000000);
            assertThat(Files.readAllBytes(destination)).isEqualTo(png);
            assertThatThrownBy(() -> reader.copy(media.id(), OTHER, destination, 1000000)).isInstanceOf(MediaException.class);
            assertThatThrownBy(() -> reader.copy(media.id(), OWNER, destination, 1)).isInstanceOf(MediaException.class);
        } finally { Files.deleteIfExists(destination); }
    }

    @Test
    void bindingHttpRequiresVersionsAndReturnsRealCommandReceipt() throws Exception {
        var media = upload(UUID.randomUUID().toString(), png, "接口.png", null, USER);
        long asset = asset(null, USER);
        var command = binding(ASSET, asset, 0, null, Map.of("coverUrl", media.id()));
        var response = http.perform(post("/api/media-bindings").contentType(MediaType.APPLICATION_JSON)
                .content(mapper.writeValueAsBytes(command))).andExpect(status().isOk()).andReturn();
        var payload = mapper.readTree(response.getResponse().getContentAsString()).path("data");
        assertThat(payload.path("commandId").asText()).isNotBlank();
        assertThat(payload.path("changes").get(0).path("version").intValue()).isEqualTo(1);
        ObjectNode missing = mapper.valueToTree(command);
        missing.remove("expectedVersion");
        http.perform(post("/api/media-bindings").contentType(MediaType.APPLICATION_JSON)
                .content(mapper.writeValueAsBytes(missing))).andExpect(status().isBadRequest());
    }

    private ResourceCommandService commands() { return context.getBean(ResourceCommandService.class); }
    private long asset(Long project, ResourceCommandContext actor) {
        return commands().execute(new ResourceCreateCommand(UUID.randomUUID().toString(), ASSET, project,
                mapper.createObjectNode().put("name", "绑定测试-" + UUID.randomUUID()).put("type", "character")), actor)
                .result().path("id").longValue();
    }
    private ResourceBindMediaCommand binding(CanvasValueType type, long id, int version, Integer parent, Map<String, String> fields) {
        return new ResourceBindMediaCommand(UUID.randomUUID().toString(), type, id, version, parent, fields);
    }
}
