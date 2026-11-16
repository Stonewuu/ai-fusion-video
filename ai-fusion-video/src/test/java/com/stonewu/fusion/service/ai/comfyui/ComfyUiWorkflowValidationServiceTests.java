package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiNativeClient;
import com.stonewu.fusion.service.canvas.CanvasJson;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ComfyUiWorkflowValidationServiceTests {
    private final ObjectMapper mapper = new ObjectMapper();
    private final ApiConfigService apis = mock(ApiConfigService.class);
    private final ComfyUiWorkflowService workflows = mock(ComfyUiWorkflowService.class);
    private final ComfyUiNativeClient client = mock(ComfyUiNativeClient.class);
    private final ApiConfig api = ApiConfig.builder().id(7L).platform("comfyui").status(1).build();
    private ComfyUiWorkflowValidationService validator;
    private ComfyUiWorkflowVersion version;
    private Map<String, JsonNode> definitions;

    @BeforeEach
    void setUp() throws Exception {
        validator = new ComfyUiWorkflowValidationService(apis, workflows,
                new ComfyUiWorkflowDocumentService(mapper), client, new CanvasJson(mapper));
        version = ComfyUiWorkflowVersion.builder().id(12L).workflowId(11L).workflowHash("hash")
                .apiWorkflowJson("""
                        {"1":{"class_type":"LoadImage","inputs":{"image":"example.png"}},
                         "2":{"class_type":"CheckpointLoader","inputs":{"ckpt_name":"known.safetensors"}},
                         "20":{"class_type":"H3","inputs":{"reference":["1",0],"model":["2",0]}},
                         "99":{"class_type":"SaveVideo","inputs":{"video":["20",0]}}}
                        """)
                .inputBindingsJson("""
                        {"referenceImages":[{"nodeId":"1","inputName":"image","valueType":"uploaded_image","index":0,"optional":true}]}
                        """)
                .outputBindingsJson("[{\"nodeId\":\"99\",\"mediaType\":\"video\",\"role\":\"primary\"}]")
                .build();
        definitions = Map.of(
                "LoadImage", mapper.readTree("{\"input\":{\"required\":{\"image\":[[\"server.png\"]]}}}"),
                "CheckpointLoader", mapper.readTree("{\"input\":{\"required\":{\"ckpt_name\":[[\"known.safetensors\"]]}}}"),
                "H3", mapper.readTree("{\"input\":{\"required\":{\"model\":[\"MODEL\"]},\"optional\":{\"reference\":[\"IMAGE\"]}}}"),
                "SaveVideo", mapper.readTree("{\"input\":{\"required\":{\"video\":[\"VIDEO\"]}}}"));
        when(apis.getById(7L)).thenReturn(api);
        when(workflows.requireVersion(12L)).thenReturn(version);
        when(workflows.requireWorkflow(11L)).thenReturn(ComfyUiWorkflow.builder().id(11L).modelType(3).apiConfigId(7L).build());
        when(client.getNodeInfo(eq(api), anyString())).thenAnswer(invocation -> {
            String type = invocation.getArgument(1);
            return mapper.createObjectNode().set(type, definitions.get(type));
        });
    }

    @Test
    void optionalUploadedExampleFileDoesNotNeedToExistOnTheTargetServer() {
        var result = validator.validateVersion(12L);
        assertThat(result.isValid()).isTrue();
        assertThat(result.getInvalidModelInputs()).isEmpty();
    }

    @Test
    void nonOptionalFileAndModelChoicesStillHaveToExist() {
        version.setInputBindingsJson(version.getInputBindingsJson().replace("\"optional\":true", "\"optional\":false"));
        var required = validator.validateVersion(12L);
        assertThat(required.isValid()).isFalse();
        assertThat(required.getInvalidModelInputs()).contains("1.image=example.png");

        version.setInputBindingsJson(version.getInputBindingsJson().replace("\"optional\":false", "\"optional\":true"));
        version.setApiWorkflowJson(version.getApiWorkflowJson().replace("known.safetensors", "missing.safetensors"));
        var model = validator.validateVersion(12L);
        assertThat(model.isValid()).isFalse();
        assertThat(model.getInvalidModelInputs()).contains("2.ckpt_name=missing.safetensors");
    }

    @Test
    void rejectsOptionalLoaderWhoseConsumerRequiresTheConnection() {
        ObjectNode h3 = (ObjectNode) definitions.get("H3");
        ((ObjectNode) h3.path("input").path("required")).set("reference", h3.path("input").path("optional").get("reference"));
        ((ObjectNode) h3.path("input").path("optional")).remove("reference");
        var result = validator.validateVersion(12L);
        assertThat(result.isValid()).isFalse();
        assertThat(result.getInvalidModelInputs()).anyMatch(value -> value.contains("20.reference") && value.contains("必填"));
    }

    @Test
    void sharedLoaderRetainsItsFileValidationWhenOnlyOneConsumerIsOptional() throws Exception {
        ObjectNode graph = (ObjectNode) mapper.readTree(version.getApiWorkflowJson());
        ((ObjectNode) graph.at("/20/inputs")).putArray("first").add("1").add(0);
        ((ObjectNode) definitions.get("H3").at("/input/required")).putArray("first").add("IMAGE");
        version.setApiWorkflowJson(graph.toString());
        version.setInputBindingsJson("""
                {"referenceImages":[{"nodeId":"20","inputName":"reference","valueType":"uploaded_image","index":0,"optional":true}]}
                """);
        var result = validator.validateVersion(12L);
        assertThat(result.isValid()).isFalse();
        assertThat(result.getInvalidModelInputs()).contains("1.image=example.png");
    }

    @Test
    void aliasesOfTheSameOptionalInputAreValidatedAsOneOmission() throws Exception {
        ObjectNode graph = (ObjectNode) mapper.readTree(version.getApiWorkflowJson());
        ((ObjectNode) graph.at("/20/inputs")).putArray("second_reference").add("1").add(0);
        ((ObjectNode) definitions.get("H3").at("/input/optional")).putArray("second_reference").add("IMAGE");
        version.setApiWorkflowJson(graph.toString());
        version.setInputBindingsJson("""
                {"referenceImages":[
                  {"nodeId":"20","inputName":"reference","valueType":"uploaded_image","index":0,"optional":true},
                  {"nodeId":"20","inputName":"second_reference","valueType":"uploaded_image","index":0,"optional":true}
                ]}
                """);
        var result = validator.validateVersion(12L);
        assertThat(result.isValid()).isTrue();
        assertThat(result.getInvalidModelInputs()).isEmpty();
    }
}
