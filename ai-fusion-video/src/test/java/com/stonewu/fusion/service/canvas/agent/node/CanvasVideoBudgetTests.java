package com.stonewu.fusion.service.canvas.agent.node;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.mapper.canvas.CanvasGenerationUsageMapper;
import com.stonewu.fusion.mapper.canvas.CanvasGenerationUsageMapper.Usage;
import com.stonewu.fusion.service.ai.ModelPresetService;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.run.CanvasRunData;
import com.stonewu.fusion.service.canvas.run.CanvasRunRepository;
import com.stonewu.fusion.service.generation.GenerationKind;
import com.stonewu.fusion.service.generation.job.GenerationModelSnapshot;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Map;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class CanvasVideoBudgetTests {
    @Test void priorAutomaticRequestsConsumeFrozenMaximumAndCannotEscapeAgentBudget() throws Exception {
        var presets = new ModelPresetService(); presets.init();
        var json = new CanvasJson(new ObjectMapper());
        var model = AiModel.builder().id(7L).config(presets.getPresetConfig("doubao-seedance-2-5-260628")).build();
        var policy = mock(CanvasAgentNodePolicy.class); var job = mock(CanvasAgentNodeRepository.Job.class);
        var scope = mock(CanvasAgentNodeScope.class);
        when(scope.videoModelId()).thenReturn(7L); when(scope.maxVideoSeconds()).thenReturn(60);
        when(job.scope()).thenReturn(scope); when(job.nodeRunId()).thenReturn("node");
        when(policy.job(any(), eq(true))).thenReturn(job);
        var runs = mock(CanvasRunRepository.class); var node = mock(CanvasRunData.NodeRun.class); var plan = mock(CanvasRunData.NodePlan.class);
        when(runs.node("node", false)).thenReturn(node); when(node.plan()).thenReturn(plan);
        when(plan.environment()).thenReturn(json.object(Map.of("videoModelId", model, "videoModelIdApiHash", "hash")));
        var models = mock(GenerationModelSnapshot.class);
        when(models.frozen(any(), eq(GenerationKind.VIDEO), eq("hash"))).thenReturn(model);
        var usages = mock(CanvasGenerationUsageMapper.class);
        when(usages.selectNodeUsage("node", "VIDEO")).thenReturn(List.of(new Usage("{\"count\":1,\"duration\":-1}", json.write(model))));
        var service = new CanvasAgentNodeGeneration(policy, runs, models, json, usages);
        var context = ResourceCommandContext.user(new CanvasActor(42, null), "canvas");
        assertThat(service.admit(context, GenerationKind.VIDEO, model, 1, 30, null)).isEqualTo("node");
        assertThatThrownBy(() -> service.admit(context, GenerationKind.VIDEO, model, 2, 30, null)).hasMessageContaining("数量上限");
    }
}
