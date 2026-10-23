package com.stonewu.fusion.service.ai.comfyui;

import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.controller.ai.AiModelController;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.ai.model.AiModelMetadataResolver;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.capability.CanvasModelCatalog;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.generation.job.GenerationModelSnapshot;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.ArrayList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** 同一工作流定义必须同时约束界面、Agent、画布和后端，并固定旧任务的能力。 */
class ComfyUiWorkflowCapabilitiesTests {
    private final ObjectMapper json = new ObjectMapper().findAndRegisterModules();
    private final ComfyUiWorkflowService workflows = mock(ComfyUiWorkflowService.class);
    private final AiModelService models = mock(AiModelService.class);
    private final ApiConfigService apis = mock(ApiConfigService.class);
    private final ModelPresetService presets = mock(ModelPresetService.class);
    private final AiModelMetadataResolver metadata = new AiModelMetadataResolver(apis);
    private final ComfyUiWorkflowCapabilities resolver = new ComfyUiWorkflowCapabilities(workflows,
            new ComfyUiWorkflowDocumentService(json));
    private final GenerationModelCapabilityService capabilities = new GenerationModelCapabilityService(
            metadata, presets, mock(ReferenceImageTransportService.class), resolver);
    private final AiModel model = AiModel.builder().id(1L).apiConfigId(10L).modelType(2).modelProtocol("comfyui")
            .comfyuiWorkflowId(12L).name("多图工作流").code("local-image").status(1).maxConcurrency(1).config("{}").build();
    private final ComfyUiWorkflow workflow = ComfyUiWorkflow.builder().id(12L).apiConfigId(10L)
            .modelType(2).status(1).activeVersionId(20L).build();

    @BeforeEach
    void setup() {
        when(workflows.requireWorkflow(12L)).thenReturn(workflow);
        when(models.getById(1L)).thenReturn(model);
        when(models.getEnabledList()).thenReturn(List.of(model));
        when(apis.getById(10L)).thenReturn(ApiConfig.builder().id(10L).platform("comfyui")
                .imageProtocol("comfyui").status(1).build());
        when(workflows.requireVersion(20L)).thenReturn(version(20L, 2));
        when(workflows.requireVersion(21L)).thenReturn(version(21L, 4));
        when(workflows.getVersions(12L)).thenReturn(List.of(version(21L, 4), version(20L, 2)));
    }

    @Test
    void frontendAgentCanvasAndSubmissionUseTheSamePublishedInputs() {
        workflow.setActiveVersionId(21L);
        var controller = new AiModelController(models, presets, capabilities, json);
        var catalog = new CanvasModelCatalog(models, apis, capabilities, workflows, json);

        assertThat(controller.list().getData().getFirst().getGenerationConfig().get("maxReferenceImages")).isEqualTo(4);
        assertThat(capabilities.buildImageCapabilitySnapshot(model).getInt("maxReferenceImages")).isEqualTo(4);
        assertThat(catalog.require(1L).versions().getFirst().inputs().get("referenceImages").maximum()).isEqualTo(4);
        capabilities.validateImageTask(model, image(4));
        assertThatThrownBy(() -> capabilities.validateImageTask(model, image(5)))
                .isInstanceOf(BusinessException.class).hasMessageContaining("最多支持 4 张");
        assertThat(model.getConfig()).isEqualTo("{}");
    }

    @Test
    void newPublicationChangesNewReadsButDoesNotChangeFrozenTasks() {
        var snapshots = new GenerationModelSnapshot(models, apis, metadata, capabilities, new CanvasJson(json));
        var frozen = snapshots.freeze(1L, GenerationKind.IMAGE);
        workflow.setActiveVersionId(21L);

        assertThat(capabilities.resolveImageCapability(model).maxReferenceImages()).isEqualTo(4);
        assertThat(capabilities.resolveImageCapability(frozen).maxReferenceImages()).isEqualTo(2);
        assertThat(JSONUtil.parseObj(frozen.getConfig()).getLong(ComfyUiWorkflowCapabilities.VERSION_KEY)).isEqualTo(20L);
        assertThat(frozen.getGenerationConfigSnapshot()).isTrue();
    }

    @Test
    void explicitlySelectedPublishedVersionDeterminesTheFrozenCapability() {
        workflow.setActiveVersionId(21L);
        var snapshots = new GenerationModelSnapshot(models, apis, metadata, capabilities, new CanvasJson(json));
        var frozen = snapshots.freeze(1L, GenerationKind.IMAGE, 20L);
        assertThat(capabilities.resolveImageCapability(frozen).maxReferenceImages()).isEqualTo(2);
        assertThat(JSONUtil.parseObj(frozen.getConfig()).getLong(ComfyUiWorkflowCapabilities.VERSION_KEY)).isEqualTo(20L);
    }

    @Test
    void historicalServerSnapshotKeepsItsLimitsWhenRestoredWithoutTheNewMarker() {
        workflow.setActiveVersionId(21L);
        var snapshots = new GenerationModelSnapshot(models, apis, metadata, capabilities, new CanvasJson(json));
        var historical = json.convertValue(model, AiModel.class);
        historical.setConfig("{\"supportReferenceImages\":true,\"maxReferenceImages\":2}");
        var restored = snapshots.frozen(historical, GenerationKind.IMAGE, snapshots.fingerprint(apis.getById(10L)));
        assertThat(capabilities.resolveImageCapability(restored).maxReferenceImages()).isEqualTo(2);
        assertThat(capabilities.resolveImageCapability(model).maxReferenceImages()).isEqualTo(4);
    }

    @Test
    void preservesExplicitListLimitAndAllowsFollowingTheWorkflowWithoutWritingRecords() {
        var listVersion = version(20L, 1);
        listVersion.setInputBindingsJson("""
                {"referenceImages":[{"nodeId":"1","inputName":"image","valueType":"string_list"}]}
                """);
        when(workflows.requireVersion(20L)).thenReturn(listVersion);
        model.setConfig("{\"maxReferenceImages\":2}");
        assertThat(capabilities.resolveImageCapability(model).maxReferenceImages()).isEqualTo(2);
        assertThat(model.getConfig()).isEqualTo("{\"maxReferenceImages\":2}");

        model.setConfig("{\"maxReferenceImages\":2,\"comfyuiInputLimits\":{\"referenceImages\":null}}");
        assertThat(capabilities.resolveImageCapability(model).maxReferenceImages()).isNull();
        var response = new AiModelController(models, presets, capabilities, json).get(1L).getData();
        var serialized = json.valueToTree(response);
        assertThat(serialized.path("generationConfig").path("maxReferenceImages").isNull()).isTrue();
    }

    @Test
    void retainsDisableRestrictionUntilAdministratorExplicitlyChoosesFollowWorkflow() {
        model.setConfig("{\"supportReferenceImages\":false,\"maxReferenceImages\":2}");
        assertThat(capabilities.resolveImageCapability(model).supportsReferenceImages()).isFalse();
        model.setConfig("{\"supportReferenceImages\":false,\"maxReferenceImages\":2,\"comfyuiInputLimits\":{\"referenceImages\":null}}");
        assertThat(capabilities.resolveImageCapability(model).supportsReferenceImages()).isTrue();
    }

    @Test
    void reportsConflictInsteadOfRaisingAUsersLimitToFitFixedSlots() {
        workflow.setActiveVersionId(21L);
        model.setConfig("{\"maxReferenceImages\":2}");
        assertThatThrownBy(() -> capabilities.getMergedModelConfig(model))
                .isInstanceOf(BusinessException.class).hasMessageContaining("使用上限 2 小于工作流所需的 4");
        var response = new AiModelController(models, presets, capabilities, json).get(1L).getData();
        assertThat(response.getCapabilityError()).contains("使用上限 2");
        assertThat(response.getGenerationConfig()).isEmpty();
        assertThat(response.getConfig()).isEqualTo("{\"maxReferenceImages\":2}");
    }

    @Test
    void declaredModelFlagsCannotEnableInputsThatTheWorkflowDoesNotBind() {
        model.setConfig("{\"supportFirstFrame\":true,\"supportLastFrame\":true,\"maxReferenceImages\":8}");
        JSONObject result = capabilities.getMergedModelConfig(model);
        assertThat(result.getBool("supportFirstFrame")).isFalse();
        assertThat(result.getBool("supportLastFrame")).isFalse();
        assertThat(result.getInt("maxReferenceImages")).isEqualTo(2);
    }

    @Test
    void cannotResolveAnUnpublishedOrForeignVersion() {
        var draft = version(21L, 4);
        draft.setPublished(false);
        when(workflows.requireVersion(21L)).thenReturn(draft);
        assertThatThrownBy(() -> capabilities.getMergedModelConfig(model, 21L))
                .isInstanceOf(BusinessException.class).hasMessageContaining("未发布");
        draft.setPublished(true);
        draft.setWorkflowId(99L);
        assertThatThrownBy(() -> capabilities.getMergedModelConfig(model, 21L))
                .isInstanceOf(BusinessException.class).hasMessageContaining("不属于");
    }

    @Test
    void conflictingOldVersionDoesNotHideOtherPublishedVersions() {
        model.setConfig("{\"comfyuiInputLimits\":{\"referenceImages\":2}}");
        var catalog = new CanvasModelCatalog(models, apis, capabilities, workflows, json);
        var option = catalog.require(1L);
        assertThat(option.versions()).hasSize(2);
        assertThat(option.versions().getFirst().unavailable()).contains("使用上限 2");
        assertThat(CanvasModelCatalog.forVersion(option, 20L).inputs().get("referenceImages").maximum()).isEqualTo(2);
        assertThatThrownBy(() -> CanvasModelCatalog.forVersion(option, 21L)).hasMessageContaining("使用上限 2");
    }

    @Test
    void agentResponseResolvesOneVersionIncludingItsSummary() {
        var publishedNext = ComfyUiWorkflow.builder().id(12L).apiConfigId(10L).modelType(2).status(1)
                .activeVersionId(21L).build();
        when(workflows.requireWorkflow(12L)).thenReturn(workflow, publishedNext);
        var result = capabilities.buildImageCapabilitySnapshot(model);
        assertThat(result.getLong(ComfyUiWorkflowCapabilities.VERSION_KEY)).isEqualTo(20L);
        assertThat(result.getJSONObject(ComfyUiWorkflowCapabilities.INPUTS_KEY)
                .getJSONObject("referenceImages").getInt("minimumWhenProvided")).isEqualTo(2);
        assertThat(result.getStr("summary")).contains("最多 2 张");
    }

    @Test
    void explicitTotalLimitAndFixedSlotsApplyToVideoInputsBeforeSubmission() {
        model.setModelType(3);
        workflow.setModelType(3);
        var video = version(20L, 1);
        video.setApiWorkflowJson("""
                {"1":{"class_type":"VideoInputs","inputs":{"images":[],"first":"default.png","a":"","b":""}}}
                """);
        video.setInputBindingsJson("""
                {"referenceImages":[{"nodeId":"1","inputName":"images","valueType":"string_list"}],
                 "firstFrame":[{"nodeId":"1","inputName":"first","valueType":"uploaded_image"}],
                 "referenceVideos":[{"nodeId":"1","inputName":"a","valueType":"string","index":0},
                                    {"nodeId":"1","inputName":"b","valueType":"string","index":1}]}
                """);
        when(workflows.requireVersion(20L)).thenReturn(video);
        model.setConfig("{\"maxImages\":2,\"maxRefVideos\":2}");
        var task = VideoTask.builder().firstFrameImageUrl("https://example.test/first.png")
                .referenceImageUrls("[\"https://example.test/a.png\",\"https://example.test/b.png\"]").build();
        assertThatThrownBy(() -> capabilities.validateVideoTask(model, task)).hasMessageContaining("最多支持 2 张图片输入");
        task.setReferenceImageUrls("[]");
        task.setReferenceVideoUrls("[\"https://example.test/a.mp4\"]");
        assertThatThrownBy(() -> capabilities.validateVideoTask(model, task)).hasMessageContaining("需传入至少 2 项");
        task.setReferenceVideoUrls("[\"https://example.test/a.mp4\",\"https://example.test/b.mp4\"]");
        capabilities.validateVideoTask(model, task);
        var resolved = capabilities.getMergedModelConfig(model);
        assertThat(resolved.getInt("maxImageInputs")).isEqualTo(2);
        assertThat(resolved.getBool("supportLastFrame")).isFalse();
    }

    @Test
    void rejectsMalformedPoliciesAndMissingIndicesWithoutInventingSlots() {
        for (String config : List.of("{\"comfyuiInputLimits\":{\"referenceImages\":-1}}",
                "{\"comfyuiInputLimits\":{\"referenceImages\":1.5}}", "{\"comfyuiInputLimits\":{\"unknown\":1}}",
                "{\"comfyuiInputLimits\":\"invalid\"}", "{broken json")) {
            model.setConfig(config);
            assertThatThrownBy(() -> capabilities.getMergedModelConfig(model)).isInstanceOf(BusinessException.class);
        }
        model.setConfig("{}");
        var gapped = version(20L, 2);
        gapped.setInputBindingsJson(gapped.getInputBindingsJson().replace("\"index\":1", "\"index\":2"));
        when(workflows.requireVersion(20L)).thenReturn(gapped);
        assertThatThrownBy(() -> capabilities.getMergedModelConfig(model)).hasMessageContaining("索引必须从 0 连续绑定");
    }

    private ImageTask image(int count) {
        var refs = new ArrayList<String>();
        for (int index = 0; index < count; index++) {
            refs.add("https://example.test/reference-" + index + ".png");
        }
        return ImageTask.builder().refImageUrls(JSONUtil.toJsonStr(refs)).build();
    }

    private ComfyUiWorkflowVersion version(long id, int count) {
        var graph = json.createObjectNode();
        var binding = json.createObjectNode();
        var refs = binding.putArray("referenceImages");
        for (int index = 0; index < count; index++) {
            String node = Integer.toString(index + 1);
            graph.putObject(node).put("class_type", "LoadImage").putObject("inputs").put("image", "default.png");
            refs.addObject().put("nodeId", node).put("inputName", "image").put("valueType", "uploaded_image")
                    .put("index", index);
        }
        return ComfyUiWorkflowVersion.builder().id(id).workflowId(12L).versionNo((int) id - 19)
                .published(true).apiWorkflowJson(graph.toString()).inputBindingsJson(binding.toString()).build();
    }
}
