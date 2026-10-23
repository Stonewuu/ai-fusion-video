package com.stonewu.fusion.service.ai.comfyui;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiNativeClient;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;

/** 保护实际提交合同：多角色素材全部绑定，配置不足时在上传和生成前拒绝。 */
class ComfyUiReferenceBindingTests {
    private final ObjectMapper json = new ObjectMapper();
    private final ComfyUiWorkflowDocumentService documents = new ComfyUiWorkflowDocumentService(json);
    private final ComfyUiInputResourceService resources = mock(ComfyUiInputResourceService.class);
    private final ComfyUiNativeClient client = mock(ComfyUiNativeClient.class);
    private final ComfyUiGenerationExecutor executor = new ComfyUiGenerationExecutor(
            mock(ApiConfigService.class), mock(ComfyUiWorkflowService.class), documents,
            new ComfyUiWorkflowRenderer(json, documents), resources, mock(ComfyUiOutputResolver.class), client);
    private final ApiConfig api = ApiConfig.builder().id(1L).platform("comfyui").build();
    private final List<String> sources = List.of("father.png", "mother.png", "son.png", "room.png");

    @BeforeEach
    void preserveUploadedInputs() {
        when(resources.uploadImages(eq(api), eq("task"), eq("referenceImages"), anyList()))
                .thenAnswer(call -> call.getArgument(3));
    }

    @Test
    void preparesAllFourReferencesInOrderWithoutSubmittingGeneration() {
        when(resources.uploadImages(api, "task", "referenceImages", sources))
                .thenReturn(List.of("uploaded-father.png", "uploaded-mother.png", "uploaded-son.png", "uploaded-room.png"));

        var prepared = executor.prepare(context(0, 1, 2, 3), "task", Map.of("referenceImages", sources));

        assertThat(prepared.prompt().at("/1/inputs/image").asText()).isEqualTo("uploaded-father.png");
        assertThat(prepared.prompt().at("/2/inputs/image").asText()).isEqualTo("uploaded-mother.png");
        assertThat(prepared.prompt().at("/3/inputs/image").asText()).isEqualTo("uploaded-son.png");
        assertThat(prepared.prompt().at("/4/inputs/image").asText()).isEqualTo("uploaded-room.png");
        verifyNoInteractions(client);
    }

    @Test
    void rejectsFourReferencesWithOnlyTwoBoundSlotsBeforeUploading() {
        assertThatThrownBy(() -> executor.prepare(context(0, 1), "task", Map.of("referenceImages", sources)))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("referenceImages")
                .hasMessageContaining("索引 2")
                .hasMessageContaining("未绑定");
        verifyNoInteractions(resources, client);
    }

    @Test
    void rejectsGapsInsteadOfInferringCapacityFromTheHighestIndex() {
        assertThatThrownBy(() -> executor.prepare(context(0, 2, 3), "task", Map.of("referenceImages", sources)))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("索引 1")
                .hasMessageContaining("未绑定");
        verifyNoInteractions(resources, client);
    }

    @Test
    void rejectsUnboundReferenceFieldBeforeUploading() {
        assertThatThrownBy(() -> executor.prepare(context(), "task", Map.of("referenceImages", sources)))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("referenceImages")
                .hasMessageContaining("未绑定");
        verifyNoInteractions(resources, client);
    }

    @Test
    void singleFileBindingWithoutIndexCannotSilentlyConsumeAList() {
        var context = context(0);
        context.version().setInputBindingsJson("""
                {"referenceImages":[{"nodeId":"1","inputName":"image","valueType":"uploaded_image"}]}
                """);

        assertThatThrownBy(() -> executor.prepare(context, "task", Map.of("referenceImages", sources)))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("索引 1")
                .hasMessageContaining("未绑定");
        verifyNoInteractions(resources, client);
    }

    @Test
    void rejectsMissingRequiredSlotBeforeUploadingEarlierImages() {
        assertThatThrownBy(() -> executor.prepare(context(0, 1, 2, 3), "task",
                Map.of("referenceImages", sources.subList(0, 3))))
                .isInstanceOf(BusinessException.class)
                .hasMessageContaining("缺少索引 3");
        verifyNoInteractions(resources, client);
    }

    @Test
    void preservesExplicitWholeListBindingsForNodesThatAcceptLists() {
        var version = ComfyUiWorkflowVersion.builder().apiWorkflowJson("""
                {"1":{"class_type":"ReferenceList","inputs":{"images":[]}}}
                """).inputBindingsJson("""
                {"referenceImages":[{"nodeId":"1","inputName":"images","valueType":"string_list"}]}
                """).build();
        var context = new ComfyUiExecutionContext(AiModel.builder().modelType(2).build(), api, null, version);

        var prepared = executor.prepare(context, "task", Map.of("referenceImages", sources));

        assertThat(prepared.prompt().at("/1/inputs/images")).isEqualTo(json.valueToTree(sources));
        verifyNoInteractions(resources, client);
    }

    private ComfyUiExecutionContext context(int... indices) {
        var workflow = json.createObjectNode();
        var bindings = json.createObjectNode();
        if (indices.length > 0) {
            var references = bindings.putArray("referenceImages");
            for (int index : indices) {
                String node = Integer.toString(index + 1);
                workflow.putObject(node).put("class_type", "LoadImage")
                        .putObject("inputs").put("image", "placeholder.png");
                references.addObject().put("nodeId", node).put("inputName", "image")
                        .put("valueType", "uploaded_image").put("index", index);
            }
        } else {
            workflow.putObject("1").put("class_type", "Text").putObject("inputs").put("text", "prompt");
        }
        var version = ComfyUiWorkflowVersion.builder().apiWorkflowJson(workflow.toString())
                .inputBindingsJson(bindings.toString()).build();
        return new ComfyUiExecutionContext(AiModel.builder().modelType(2).build(), api, null, version);
    }
}
