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

    @Test
    void redirectsUploadedImageOnlyToDirectLoadImageAndPreservesVersion() {
        ComfyUiWorkflowVersion version = linkedVersion("LoadImage", 0);
        ObjectNode result = renderer.render(2, version, Map.of("referenceImages", List.of("new.png")));
        assertThat(result.at("/1/inputs/image").asText()).isEqualTo("new.png");
        assertThat(result.at("/9/inputs/images").isArray()).isTrue();
        assertThat(version.getApiWorkflowJson()).contains("old.png");
    }

    @Test
    void rejectsProcessingNodeInsteadOfOverwritingItsImageTensor() {
        assertThatThrownBy(() -> renderer.render(2, linkedVersion("ImageScale", 0),
                Map.of("referenceImages", List.of("new.png"))))
                .isInstanceOf(BusinessException.class).hasMessageContaining("加载节点的文件输入");
    }

    @Test
    void rejectsLoadImageMaskOutput() {
        assertThatThrownBy(() -> renderer.render(2, linkedVersion("LoadImage", 1),
                Map.of("referenceImages", List.of("new.png"))))
                .isInstanceOf(BusinessException.class).hasMessageContaining("IMAGE 输出");
    }

    @Test
    void rejectsDifferentImagesBoundToTheSameLoader() {
        ComfyUiWorkflowVersion version = linkedVersion("LoadImage", 0);
        version.setInputBindingsJson("""
                {"referenceImages":[
                  {"nodeId":"1","inputName":"image","valueType":"uploaded_image","index":0},
                  {"nodeId":"9","inputName":"images","valueType":"uploaded_image","index":1}
                ]}
                """);
        assertThatThrownBy(() -> renderer.render(2, version,
                Map.of("referenceImages", List.of("first.png", "last.png"))))
                .isInstanceOf(BusinessException.class).hasMessageContaining("同一个加载输入");
    }

    @Test
    void nineOptionalLoadersUseOnlyTheThreeProvidedImagesWithoutChangingTheSavedVersion() {
        var version = nineOptionalImages();
        String saved = version.getApiWorkflowJson();
        var rendered = renderer.render(3, version, Map.of("referenceImages", List.of("a.png", "b.png", "c.png")));
        assertThat(rendered.size()).isEqualTo(5);
        assertThat(rendered.at("/1/inputs/image").asText()).isEqualTo("a.png");
        assertThat(rendered.at("/3/inputs/image").asText()).isEqualTo("c.png");
        assertThat(rendered.at("/20/inputs/ref_3").isArray()).isTrue();
        assertThat(rendered.at("/20/inputs/ref_4").isMissingNode()).isTrue();
        assertThat(rendered.has("4")).isFalse();
        assertThat(rendered.toString()).doesNotContain("example-");
        assertThat(version.getApiWorkflowJson()).isEqualTo(saved);

        var textOnly = renderer.render(3, version, Map.of());
        assertThat(textOnly.size()).isEqualTo(2);
        assertThat(textOnly.at("/20/inputs/prompt").asText()).isEqualTo("a story");
        assertThat(textOnly.at("/20/inputs/ref_1").isMissingNode()).isTrue();
        assertThat(textOnly.toString()).doesNotContain("example-");
    }

    @Test
    void omitsAudioAndVideoLoadersAndRendersProvidedBuiltInLoaderLinks() {
        var version = videoVersion("""
                {"1":{"class_type":"LoadAudio","inputs":{"audio":"old.wav"}},
                 "2":{"class_type":"LoadVideo","inputs":{"file":"old.mp4"}},
                 "20":{"class_type":"H3","inputs":{"ref_audio":["1",0],"ref_video":["2",0]}},
                 "99":{"class_type":"SaveVideo","inputs":{"video":["20",0]}}}
                """, """
                {"referenceAudios":[{"nodeId":"20","inputName":"ref_audio","valueType":"uploaded_audio","index":0,"optional":true}],
                 "referenceVideos":[{"nodeId":"20","inputName":"ref_video","valueType":"uploaded_video","index":0,"optional":true}]}
                """);
        var none = renderer.render(3, version, Map.of());
        assertThat(none.size()).isEqualTo(2);
        assertThat(none.at("/20/inputs").isEmpty()).isTrue();
        var provided = renderer.render(3, version,
                Map.of("referenceAudios", List.of("new.wav"), "referenceVideos", List.of("new.mp4")));
        assertThat(provided.at("/1/inputs/audio").asText()).isEqualTo("new.wav");
        assertThat(provided.at("/2/inputs/file").asText()).isEqualTo("new.mp4");
        assertThat(provided.at("/20/inputs/ref_video").isArray()).isTrue();
    }

    @Test
    void disconnectingOneOptionalConsumerKeepsASharedLoaderNeededByAnotherInput() {
        var version = videoVersion("""
                {"1":{"class_type":"LoadImage","inputs":{"image":"old.png"}},
                 "20":{"class_type":"H3","inputs":{"first":["1",0],"reference":["1",0]}},
                 "99":{"class_type":"SaveVideo","inputs":{"video":["20",0]}}}
                """, """
                {"firstFrame":[{"nodeId":"20","inputName":"first","valueType":"uploaded_image"}],
                 "referenceImages":[{"nodeId":"20","inputName":"reference","valueType":"uploaded_image","index":0,"optional":true}]}
                """);
        var result = renderer.render(3, version, Map.of("firstFrame", "first.png"));
        assertThat(result.at("/1/inputs/image").asText()).isEqualTo("first.png");
        assertThat(result.at("/20/inputs/first").isArray()).isTrue();
        assertThat(result.at("/20/inputs/reference").isMissingNode()).isTrue();
    }

    @Test
    void refusesToRemoveALoaderUsedByAnotherBoundInput() {
        var version = videoVersion("""
                {"1":{"class_type":"LoadImage","inputs":{"image":"old.png"}},
                 "20":{"class_type":"H3","inputs":{"first":["1",0]}},
                 "99":{"class_type":"SaveVideo","inputs":{"video":["20",0]}}}
                """, """
                {"referenceImages":[{"nodeId":"1","inputName":"image","valueType":"uploaded_image","index":0,"optional":true}],
                 "firstFrame":[{"nodeId":"20","inputName":"first","valueType":"uploaded_image"}]}
                """);
        assertThatThrownBy(() -> renderer.render(3, version, Map.of("firstFrame", "first.png")))
                .isInstanceOf(BusinessException.class).hasMessageContaining("firstFrame");
    }

    @Test
    void omittingLiteralOptionalMediaOnlyRemovesTheInputAndKeepsItsConsumer() {
        var version = videoVersion("""
                {"20":{"class_type":"H3","inputs":{"reference_urls":["old.png"],"prompt":"story"}},
                 "99":{"class_type":"SaveVideo","inputs":{"video":["20",0]}}}
                """, """
                {"referenceImages":[{"nodeId":"20","inputName":"reference_urls","valueType":"string_list","optional":true}]}
                """);
        var result = renderer.render(3, version, Map.of());
        assertThat(result.has("20")).isTrue();
        assertThat(result.at("/20/inputs/reference_urls").isMissingNode()).isTrue();
        assertThat(result.at("/20/inputs/prompt").asText()).isEqualTo("story");
    }

    private ComfyUiWorkflowVersion nineOptionalImages() {
        var mapper = new ObjectMapper();
        var workflow = mapper.createObjectNode();
        var bindings = mapper.createObjectNode().putArray("referenceImages");
        var h3Inputs = workflow.putObject("20").put("class_type", "H3").putObject("inputs");
        h3Inputs.put("prompt", "a story");
        for (int number = 1; number <= 9; number++) {
            String id = String.valueOf(number);
            workflow.putObject(id).put("class_type", "LoadImage").putObject("inputs").put("image", "example-" + id + ".png");
            h3Inputs.putArray("ref_" + id).add(id).add(0);
            bindings.addObject().put("nodeId", id).put("inputName", "image").put("valueType", "uploaded_image")
                    .put("index", number - 1).put("optional", true);
        }
        workflow.putObject("99").put("class_type", "SaveVideo").putObject("inputs").putArray("video").add("20").add(0);
        var inputBindings = mapper.createObjectNode();
        inputBindings.set("referenceImages", bindings);
        return videoVersion(workflow.toString(), inputBindings.toString());
    }

    private ComfyUiWorkflowVersion videoVersion(String workflow, String bindings) {
        return ComfyUiWorkflowVersion.builder().apiWorkflowJson(workflow).inputBindingsJson(bindings)
                .outputBindingsJson("[{\"nodeId\":\"99\",\"mediaType\":\"video\",\"role\":\"primary\"}]").build();
    }

    private ComfyUiWorkflowVersion linkedVersion(String sourceClass, int slot) {
        return version("""
                {"1":{"class_type":"%s","inputs":{"image":"old.png"}},
                 "9":{"class_type":"SaveImage","inputs":{"images":["1",%d]}}}
                """.formatted(sourceClass, slot), """
                {"referenceImages":[{"nodeId":"9","inputName":"images","valueType":"uploaded_image","index":0}]}
                """);
    }
}
