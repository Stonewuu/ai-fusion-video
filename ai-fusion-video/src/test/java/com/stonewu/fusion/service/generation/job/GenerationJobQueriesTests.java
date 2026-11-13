package com.stonewu.fusion.service.generation.job;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.GenerationExecutionContexts;
import com.stonewu.fusion.service.generation.GenerationKind;
import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.*;

class GenerationJobQueriesTests {
    private final CanvasJson json = new CanvasJson(new ObjectMapper().findAndRegisterModules());
    private final GenerationJobRepository jobs = mock(GenerationJobRepository.class);
    private final GenerationExecutionContexts access = mock(GenerationExecutionContexts.class);
    private final GenerationAttemptRepository attempts = mock(GenerationAttemptRepository.class);
    private final GenerationReviewPolicy reviews = mock(GenerationReviewPolicy.class);
    private final GenerationJobQueries queries = new GenerationJobQueries(jobs, access, attempts, reviews, json);
    private final CanvasActor actor = new CanvasActor(42, null);
    private final LocalDateTime created = LocalDateTime.of(2026, 11, 12, 1, 4, 19);

    // 详情使用提交时的快照，隐藏内部上下文，并保留 0、false 和未配置参数的区别。
    @Test
    void exposesFrozenInputAndAttemptTimesWithoutInventingAcknowledgement() {
        var submission = submission();
        var job = job("job", 1);
        when(jobs.jobs(submission.id())).thenReturn(List.of(job));
        when(attempts.forJob("job")).thenReturn(List.of(new GenerationAttemptRepository.Summary(
                "attempt", 1, null, true, false, null, created, created.plusSeconds(6), null, null, null)));

        var view = queries.get("submission", actor);

        assertThat(view.input().prompt()).isEqualTo("人物转身说话");
        assertThat(view.input().parameters().path("seed").asLong()).isZero();
        assertThat(view.input().parameters().path("generateAudio").asBoolean()).isFalse();
        assertThat(view.input().parameters().path("ratio").isNull()).isTrue();
        assertThat(view.input().references()).containsExactly(new GenerationExecutionInput.Reference("image", "media://portrait"));
        assertThat(view.modelCode()).isEqualTo("gemini-omni-1.2-flash");
        assertThat(view.modelProtocol()).isEqualTo("vertex_ai");
        assertThat(json.write(view)).doesNotContain("private-key", "userId", "ownerInstanceId", "_resultItemIds", "config");
        var item = view.items().getFirst();
        assertThat(item.state()).isEqualTo("UNKNOWN_SUBMISSION");
        assertThat(item.createdAt()).isEqualTo(Instant.parse("2026-11-12T01:04:19Z"));
        assertThat(item.updatedAt()).isEqualTo(Instant.parse("2026-11-12T01:06:04Z"));
        assertThat(item.attempts().getFirst().submittedAt()).isEqualTo(Instant.parse("2026-11-12T01:04:25Z"));
        assertThat(item.attempts().getFirst().acknowledgedAt()).isNull();
        assertThat(item.attempts().getFirst().acknowledged()).isFalse();
    }

    @Test
    void requestedCountIncludesEveryResultGroup() {
        submission();
        var children = List.of(job("first", 1), job("second", 2));
        when(jobs.jobs("submission")).thenReturn(children);

        var view = queries.get("submission", actor);

        assertThat(view.input().parameters().path("count").asInt()).isEqualTo(2);
        assertThat(view.items()).hasSize(2);
    }

    @Test
    void unauthorizedReaderCannotReachInputsOrAttemptHistory() {
        submission();
        doThrow(new GenerationJobException(403, "FORBIDDEN", "无权查看生成任务"))
                .when(access).requireRead(GenerationKind.VIDEO, 12, actor);

        assertThatThrownBy(() -> queries.get("submission", actor)).hasMessageContaining("无权查看");

        verify(jobs, never()).jobs(anyString());
        verifyNoInteractions(attempts);
    }

    @Test
    void imageInputPreservesReferenceOrderAndDoesNotMutateSnapshot() {
        var input = json.read("""
                {"prompt":"雨中山门","count":1,"width":1024,"height":1024,
                 "refImageUrls":"[\\"media://first\\",\\"media://second\\"]","_resultItemIds":[1]}
                """, ObjectNode.class);

        var view = GenerationExecutionInput.from(input, 3, json);

        assertThat(view.references()).extracting(GenerationExecutionInput.Reference::source)
                .containsExactly("media://first", "media://second");
        assertThat(view.parameters().path("count").asInt()).isEqualTo(3);
        assertThat(input.path("count").asInt()).isEqualTo(1);
        assertThat(view.parameters().has("_resultItemIds")).isFalse();
    }

    private GenerationSubmission submission() {
        var submission = mock(GenerationSubmission.class);
        var model = AiModel.builder().id(26L).name("视频模型").code("gemini-omni-1.2-flash")
                .modelProtocol("vertex_ai").config("private-key").build();
        when(submission.id()).thenReturn("submission");
        when(submission.kind()).thenReturn(GenerationKind.VIDEO);
        when(submission.taskId()).thenReturn(12L);
        when(submission.publicTaskId()).thenReturn("public-task");
        when(submission.userId()).thenReturn(42L);
        when(submission.model()).thenReturn(model);
        when(submission.state()).thenReturn("FAILED");
        when(submission.createdAt()).thenReturn(created);
        when(jobs.submission("submission")).thenReturn(submission);
        return submission;
    }

    private RemoteGenerationJob job(String id, long itemId) {
        var job = mock(RemoteGenerationJob.class);
        when(job.id()).thenReturn(id);
        when(job.itemIds()).thenReturn(List.of(itemId));
        when(job.state()).thenReturn("UNKNOWN_SUBMISSION");
        when(job.phase()).thenReturn("FINISHED");
        when(job.deadline()).thenReturn(created.plusHours(1));
        when(job.createdAt()).thenReturn(created);
        when(job.updatedAt()).thenReturn(created.plusSeconds(105));
        var input = json.read("""
                {"prompt":"人物转身说话","count":1,"seed":0,"generateAudio":false,"ratio":null,
                 "referenceImageUrls":"[\\"media://portrait\\"]","userId":42,"ownerInstanceId":"private-key",
                 "_resultItemIds":[1]}
                """, ObjectNode.class);
        when(job.input()).thenReturn(input);
        return job;
    }
}
