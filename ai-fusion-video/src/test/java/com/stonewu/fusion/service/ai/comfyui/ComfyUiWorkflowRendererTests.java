package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ComfyUiWorkflowRendererTests {

    private ComfyUiWorkflowRenderer renderer;

    @BeforeEach
    void setUp() {
        ObjectMapper objectMapper = new ObjectMapper();
        renderer = new ComfyUiWorkflowRenderer(
                objectMapper,
                new ComfyUiWorkflowDocumentService(objectMapper));
    }

    @Test
    void renderWritesTypedValuesToExplicitTargets() {
        ComfyUiWorkflowVersion version = version(
                """
                        {
                          "1":{"class_type":"Text","inputs":{"text":"old"}},
                          "2":{"class_type":"Sampler","inputs":{"seed":0,"batch_size":1}},
                          "3":{"class_type":"LoadImage","inputs":{"image":"old.png"}},
                          "9":{"class_type":"SaveImage","inputs":{"images":["2",0]}}
                        }
                        """,
                """
                        {
                          "prompt":[{"nodeId":"1","inputName":"text","valueType":"string"}],
                          "seed":[{"nodeId":"2","inputName":"seed","valueType":"integer"}],
                          "count":[{"nodeId":"2","inputName":"batch_size","valueType":"integer"}],
                          "referenceImages":[{"nodeId":"3","inputName":"image","valueType":"uploaded_image","index":1}]
                        }
                        """);

        ObjectNode result = renderer.render(2, version, Map.of(
                "prompt", "new prompt",
                "seed", 42,
                "count", 3,
                "referenceImages", List.of("first.png", "second.png")));

        assertThat(result.at("/1/inputs/text").asText()).isEqualTo("new prompt");
        assertThat(result.at("/2/inputs/seed").asLong()).isEqualTo(42L);
        assertThat(result.at("/2/inputs/batch_size").asInt()).isEqualTo(3);
        assertThat(result.at("/3/inputs/image").asText()).isEqualTo("second.png");
    }

    @Test
    void renderRejectsMissingIndexedInput() {
        ComfyUiWorkflowVersion version = version(
                """
                        {
                          "3":{"class_type":"LoadImage","inputs":{"image":"old.png"}},
                          "9":{"class_type":"SaveImage","inputs":{"images":["3",0]}}
                        }
                        """,
                """
                        {"referenceImages":[{"nodeId":"3","inputName":"image","valueType":"uploaded_image","index":1}]}
                        """);

        assertThatThrownBy(() -> renderer.render(
                2, version, Map.of("referenceImages", List.of("only.png"))))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("缺少索引 1");
    }

    private ComfyUiWorkflowVersion version(String apiJson, String inputBindingsJson) {
        return ComfyUiWorkflowVersion.builder()
                .apiWorkflowJson(apiJson)
                .inputBindingsJson(inputBindingsJson)
                .outputBindingsJson(
                        "[{\"nodeId\":\"9\",\"mediaType\":\"image\",\"role\":\"primary\"}]")
                .build();
    }
}
