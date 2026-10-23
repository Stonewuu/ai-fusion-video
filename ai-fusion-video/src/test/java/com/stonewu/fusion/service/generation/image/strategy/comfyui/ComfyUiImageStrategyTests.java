package com.stonewu.fusion.service.generation.image.strategy.comfyui;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.entity.ai.ApiConfig;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiExecutionContext;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiGenerationExecutor;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiPreparedSubmission;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiStoredOutput;
import com.stonewu.fusion.service.ai.comfyui.client.ComfyUiJobResult;
import com.stonewu.fusion.service.generation.image.ImageGenerationService;
import com.stonewu.fusion.service.generation.job.GenerationLeaseScope;
import com.stonewu.fusion.service.generation.job.RemoteGenerationJob;
import com.stonewu.fusion.service.generation.job.RemoteGenerationResult;
import com.stonewu.fusion.service.media.GenerationMediaService;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

class ComfyUiImageStrategyTests {
    private final ComfyUiGenerationExecutor executor = mock(ComfyUiGenerationExecutor.class);
    private final ImageGenerationService records = mock(ImageGenerationService.class);
    private final AiModelService models = mock(AiModelService.class);
    private final ComfyUiImageStrategy strategy = new ComfyUiImageStrategy(models, records, executor, mock(GenerationMediaService.class));
    private final AiModel model = AiModel.builder().id(5L).modelType(2).build();
    private final ApiConfig api = ApiConfig.builder().id(7L).build();
    private final ImageTask task = ImageTask.builder().id(10L).modelId(5L).workflowVersionId(20L)
            .taskId("task").prompt("test").count(2).width(512).height(512).build();
    private final ComfyUiExecutionContext context = new ComfyUiExecutionContext(model, api, null, null);
    private final RemoteGenerationJob.Claim claim = new RemoteGenerationJob.Claim(UUID.randomUUID().toString(), "worker", 1);

    @Test void preparesStableRemoteIdAndBatchBeforeSendingAnyGenerationRequest() {
        when(executor.resolveContext(model, 20L, api)).thenReturn(context);
        var prepared = new ComfyUiPreparedSubmission(context, UUID.randomUUID().toString(), new ObjectMapper().createObjectNode());
        when(executor.prepare(eq(context), eq(claim.id()), any(Map.class))).thenReturn(prepared);
        try (var ignored = GenerationLeaseScope.enter(claim)) {
            assertThat(strategy.batchSize(task, model, api)).isEqualTo(2);
            var request = strategy.prepare(task, model, api);
            verify(executor, never()).submit(any());
            assertThat(request.intent().providerTaskId()).isEqualTo(prepared.promptId());
            var receipt = request.submit();
            assertThat(receipt.providerTaskId()).isEqualTo(prepared.promptId());
            verify(executor).submit(prepared);
            verifyNoInteractions(records, models);
        }
    }

    @Test void onePendingQueryReturnsWithoutWaitingOrWritingBusinessItems() {
        when(executor.resolveContext(model, 20L, api)).thenReturn(context);
        String id = UUID.randomUUID().toString();
        when(executor.query(context, id)).thenReturn(new ComfyUiJobResult(id, "in_progress", new ObjectMapper().createObjectNode(), null, null));
        var result = strategy.query(id, task, model, api);
        assertThat(result.state()).isEqualTo(RemoteGenerationResult.State.PENDING);
        verify(executor).resolveContext(model, 20L, api);
        verify(executor).query(context, id);
        verifyNoMoreInteractions(executor);
        verifyNoInteractions(records, models);
    }

    @Test void completedQueryReturnsNamedMediaWithoutPublishingPrematurely() {
        when(executor.resolveContext(model, 20L, api)).thenReturn(context);
        String id = UUID.randomUUID().toString();
        String media = "media://" + UUID.randomUUID();
        String cover = "media://" + UUID.randomUUID();
        var remote = new ComfyUiJobResult(id, "completed", new ObjectMapper().createObjectNode(), null, null);
        when(executor.query(context, id)).thenReturn(remote);
        when(executor.storeOutputs(eq(context), eq(remote), any())).thenReturn(List.of(new ComfyUiStoredOutput("image", "primary", media, 123L)));
        try (var ignored = GenerationLeaseScope.enter(claim)) {
            var result = strategy.query(id, task, model, api);
            assertThat(result.state()).isEqualTo(RemoteGenerationResult.State.SUCCEEDED);
            assertThat(result.outputs()).hasSize(1);
            assertThat(result.outputs().getFirst().media()).isEqualTo(media);
            verifyNoInteractions(records, models);
        }
    }
}
