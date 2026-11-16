package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ComfyUiWorkflowDocumentServiceTests {

    private ComfyUiWorkflowDocumentService documentService;

    @BeforeEach
    void setUp() {
        documentService = new ComfyUiWorkflowDocumentService(new ObjectMapper());
    }

    @Test
    void normalizeCanonicalizesApiWorkflowAndBindings() {
        ComfyUiWorkflowDefinition definition = documentService.normalize(
                2,
                null,
                """
                        {
                          "9": {"class_type":"SaveImage","inputs":{"images":["8",0]}},
                          "1": {"class_type":"CLIPTextEncode","inputs":{"text":"old"}}
                        }
                        """,
                """
                        {"prompt":[{"nodeId":"1","inputName":"text","valueType":"string"}]}
                        """,
                """
                        [{"nodeId":"9","mediaType":"image","role":"primary"}]
                        """);

        assertThat(definition.apiWorkflowJson()).startsWith("{\"1\":");
        assertThat(definition.requiredNodesJson()).isEqualTo("[\"CLIPTextEncode\",\"SaveImage\"]");
        assertThat(definition.workflowHash()).hasSize(64);
        assertThat(definition.inputBindings()).containsExactly(
                new ComfyUiInputBinding("prompt", "1", "text", "string", null));
        assertThat(definition.outputBindings()).containsExactly(
                new ComfyUiOutputBinding("9", "image", "primary"));
    }

    @Test
    void normalizeRejectsUiFormatInsteadOfGuessing() {
        assertThatThrownBy(() -> documentService.normalize(
                2, null, "{\"nodes\":[],\"links\":[]}", "{}", "[]"))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("UI-format");
    }

    @Test
    void normalizeRejectsEmbeddedPlaintextSecret() {
        assertThatThrownBy(() -> documentService.normalize(
                2,
                null,
                """
                        {"1":{"class_type":"RemoteNode","inputs":{"api_key":"secret"}}}
                        """,
                "{}",
                "[{\"nodeId\":\"1\",\"mediaType\":\"image\",\"role\":\"primary\"}]"))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("明文密钥");
    }

    @Test
    void preservesOptionalMediaAndRejectsMalformedBindingFlags() {
        String workflow = "{\"1\":{\"class_type\":\"LoadImage\",\"inputs\":{\"image\":\"example.png\"}}}";
        String binding = "{\"referenceImages\":[{\"nodeId\":\"1\",\"inputName\":\"image\",\"valueType\":\"uploaded_image\",\"index\":0,\"optional\":true}]}";
        assertThat(documentService.parseInputBindings(2, workflow, binding).getFirst().optional()).isTrue();
        assertThat(documentService.parseInputBindings(2, workflow, binding.replace(",\"optional\":true", ""))
                .getFirst().optional()).isFalse();
        assertThatThrownBy(() -> documentService.parseInputBindings(2, workflow,
                binding.replace("\"optional\":true", "\"optional\":\"true\"")))
                .hasMessageContaining("optional 必须为布尔值");
        assertThatThrownBy(() -> documentService.parseInputBindings(2, workflow,
                binding.replace("\"index\":0", "\"index\":0.5")))
                .hasMessageContaining("index 必须是非负整数");
        assertThatThrownBy(() -> documentService.parseInputBindings(2, workflow,
                binding.replace("referenceImages", "prompt"))).hasMessageContaining("只有媒体输入可以设置为可选");
    }

    @Test
    void normalizeRequiresPrimaryOutputMatchingModelType() {
        assertThatThrownBy(() -> documentService.normalize(
                3,
                null,
                "{\"1\":{\"class_type\":\"SaveVideo\",\"inputs\":{}}}",
                "{}",
                "[{\"nodeId\":\"1\",\"mediaType\":\"image\",\"role\":\"primary\"}]"))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("primary video");
    }
}
