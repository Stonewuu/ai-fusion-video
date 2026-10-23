package com.stonewu.fusion.service.canvas.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasGraph;
import com.stonewu.fusion.service.canvas.model.CanvasNode;
import com.stonewu.fusion.service.canvas.model.CanvasNodeKind;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.generation.job.GenerationSubmission;
import com.stonewu.fusion.service.generation.job.GenerationSubmissionService;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;
import java.util.List;
import java.util.Map;
import static com.stonewu.fusion.service.canvas.run.CanvasRunData.*;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class CanvasAutomaticVideoTests {
    // 未选时长时使用模型默认值；自动时长按上限计入预算，提交时仍保留 -1。
    @ParameterizedTest
    @CsvSource({", 5, 10", "-1, -1, 60", "4, 4, 8"})
    void durationSelectionControlsBudgetAndActualRequest(Integer selectedDuration, int requestDuration, int budgetSeconds) {
        var presets = new ModelPresetService();
        presets.init();
        var mapper = new ObjectMapper();
        var json = new CanvasJson(mapper);
        var model = AiModel.builder().id(7L).code("doubao-seedance-2-5-260628")
                .config(presets.getPresetConfig("doubao-seedance-2-5-260628")).build();
        var config = mapper.createObjectNode().put("modelId", 7).put("count", 2);
        if (selectedDuration != null) {
            config.put("duration", selectedDuration);
        }
        var node = mock(CanvasNode.class);
        when(node.kind()).thenReturn(CanvasNodeKind.VIDEO_GENERATION);
        when(node.config()).thenReturn(config);
        var environment = mapper.createObjectNode();
        environment.set("model", json.object(model));
        environment.put("apiHash", "frozen-api");
        environment.putObject("capability").set("parameters", mapper.valueToTree(presets.getPreset(model.getCode()).getJSONObject("config")));
        var budgeting = mock(CanvasRunEnvironment.class, CALLS_REAL_METHODS);
        assertThat(budgeting.seconds(node, environment)).isEqualTo(budgetSeconds);
        var plan = new NodePlan(node, "hash", "EXECUTE", null, List.of(), List.of(), Map.of(), Ports.empty(), environment, 0, budgetSeconds);
        assertThat(CanvasRunInputs.optionalVideoPrompt(plan)).isTrue();

        var nodeRun = mock(NodeRun.class);
        when(nodeRun.id()).thenReturn("node");
        when(nodeRun.plan()).thenReturn(plan);
        when(nodeRun.inputs()).thenReturn(new Ports(Map.of("firstFrame", List.of(new Value("frame", CanvasValueType.IMAGE,
                null, "00000000-0000-4000-8000-000000000001", 1, null, null, null)))));
        var run = mock(Run.class);
        var runPlan = mock(Plan.class);
        when(run.actor()).thenReturn(new CanvasActor(42, null));
        when(run.canvasId()).thenReturn("canvas");
        when(run.plan()).thenReturn(runPlan);
        when(runPlan.graph()).thenReturn(new CanvasGraph(1, null, List.of(), List.of()));
        var submissions = mock(GenerationSubmissionService.class);
        var accepted = mock(GenerationSubmission.class);
        when(accepted.id()).thenReturn("submission");
        when(submissions.videoFrozen(any(), anyString(), any(), anyString(), any(), anyString())).thenReturn(accepted);
        var executor = new CanvasGenerationExecutor(submissions, null, null, null, json, null, null);
        executor.step(run, nodeRun, null);
        var request = ArgumentCaptor.forClass(VideoTask.class);
        verify(submissions).videoFrozen(request.capture(), anyString(), any(), eq("node"), any(), eq("frozen-api"));
        assertThat(request.getValue().getDuration()).isEqualTo(requestDuration);
        assertThat(request.getValue().getPrompt()).isEmpty();
        assertThat(request.getValue().getInputMediaVersions()).containsEntry("00000000-0000-4000-8000-000000000001", 1);
    }
}
