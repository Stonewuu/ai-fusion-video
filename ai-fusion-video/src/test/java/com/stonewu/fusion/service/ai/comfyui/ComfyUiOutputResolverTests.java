package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiJobResult;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ComfyUiOutputResolverTests {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final ComfyUiOutputResolver resolver = new ComfyUiOutputResolver();

    @Test
    void resolvesOfficialFileDescriptorFromExplicitlyBoundNode() throws Exception {
        ObjectNode outputs = (ObjectNode) objectMapper.readTree(
                "{\"9\":{\"images\":[{\"filename\":\"result.png\","
                        + "\"subfolder\":\"final\",\"type\":\"output\"}]}}");
        ComfyUiJobResult job = new ComfyUiJobResult(
                "job", "completed", outputs, null, objectMapper.createObjectNode());

        assertThat(resolver.resolve(job,
                List.of(new ComfyUiOutputBinding("9", "image", "primary"))))
                .containsExactly(new ComfyUiRemoteOutput(
                        "9", "image", "primary", "result.png", "final", "output"));
    }

    @Test
    void rejectsFileDescriptorMissingOfficialTypeField() throws Exception {
        ObjectNode outputs = (ObjectNode) objectMapper.readTree(
                "{\"9\":{\"images\":[{\"filename\":\"result.png\",\"subfolder\":\"\"}]}}");
        ComfyUiJobResult job = new ComfyUiJobResult(
                "job", "completed", outputs, null, objectMapper.createObjectNode());

        assertThatThrownBy(() -> resolver.resolve(job,
                List.of(new ComfyUiOutputBinding("9", "image", "primary"))))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("filename、subfolder、type");
    }
}
