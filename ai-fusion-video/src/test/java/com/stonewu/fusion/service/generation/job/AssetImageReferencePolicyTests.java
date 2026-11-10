package com.stonewu.fusion.service.generation.job;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.ai.AgentRun;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.mapper.ai.AgentRunMapper;
import com.stonewu.fusion.mapper.generation.GenerationSubmissionMapper;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.resource.model.ResourceCommandContext;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/** 保护失败重试与恢复执行时的角色、画风引用，避免重新提交时静默变成纯文生图。 */
class AssetImageReferencePolicyTests {
    private static final String STYLE = "http://localhost:3000/api/art-styles/realistic.jpg";
    private static final String CHARACTER_ID = "00000000-0000-0000-0000-000000000001";
    private static final String CHARACTER = "media://" + CHARACTER_ID;
    private static final String OTHER = "media://00000000-0000-0000-0000-000000000002";
    private static final CanvasJson JSON = new CanvasJson(new ObjectMapper().findAndRegisterModules());
    private final AgentRunMapper runs = mock(AgentRunMapper.class);
    private final GenerationSubmissionMapper submissions = mock(GenerationSubmissionMapper.class);
    private final GenerationRequestRepository requests = mock(GenerationRequestRepository.class);
    private final ReferenceImageTransportService references = new ReferenceImageTransportService(null, null, null, null);
    private final ResourceCommandContext context = new ResourceCommandContext(new CanvasActor(9, null),
            "agent", null, "asset-run", "retry-call", "restored-worker", 2L);

    private AssetImageReferencePolicy policy() {
        return new AssetImageReferencePolicy(runs, submissions, requests, references, JSON);
    }

    private void assetRun() {
        when(runs.selectByRunId(context.agentRunId())).thenReturn(AgentRun.builder()
                .agentType("asset_image_executor").deadlineAt(LocalDateTime.now().plusMinutes(10)).build());
        when(requests.scope(context)).thenReturn("scope");
    }

    private void pinned(String urls, Map<String, Integer> versions) {
        assetRun();
        var input = ImageTask.builder().prompt("首轮输入").refImageUrls(urls).inputMediaVersions(versions).build();
        var stored = JSON.object(input);
        stored.putArray("_resultItemIds").add(42);
        when(submissions.selectAssetImageReferenceSnapshot("scope", "asset-run"))
                .thenReturn(new AssetImageReferenceSnapshot("first-submission", JSON.write(stored)));
    }

    static Stream<String> changedReferences() {
        return Stream.of(null, "[]", JSON.write(List.of(STYLE)), JSON.write(List.of(CHARACTER)),
                JSON.write(List.of(CHARACTER, STYLE)), JSON.write(List.of(STYLE, OTHER)),
                JSON.write(List.of(STYLE, CHARACTER, OTHER)));
    }

    @ParameterizedTest
    @MethodSource("changedReferences")
    void rejectsOmittedEmptyRemovedReplacedReorderedOrAddedReferences(String urls) {
        pinned(JSON.write(List.of(STYLE, CHARACTER)), Map.of(CHARACTER_ID, 3));
        var retry = ImageTask.builder().prompt("第二轮调整提示词").refImageUrls(urls).build();

        assertThatThrownBy(() -> policy().enforce(retry, context))
                .isInstanceOfSatisfying(GenerationJobException.class,
                        failure -> assertThat(failure.getErrorCode()).isEqualTo("ASSET_IMAGE_REFERENCES_CHANGED"));
    }

    @Test
    void restoredExecutionRetainsReferencesAndTheOriginalMediaVersion() {
        pinned(JSON.write(List.of(STYLE, CHARACTER)), Map.of(CHARACTER_ID, 3));
        var retry = ImageTask.builder().prompt("修改后的合法提示词")
                .refImageUrls(JSON.write(List.of(STYLE, CHARACTER))).build();

        policy().enforce(retry, context);

        assertThat(retry.getInputMediaVersions()).containsExactlyEntriesOf(Map.of(CHARACTER_ID, 3));
        assertThat(retry.getPrompt()).isEqualTo("修改后的合法提示词");
        assertThat(references.parseJsonInputs(retry.getRefImageUrls(), "refImageUrls"))
                .containsExactly(STYLE, CHARACTER);
    }

    @Test
    void cannotReplaceTheOriginalMediaVersionWithANewerOne() {
        pinned(JSON.write(List.of(CHARACTER)), Map.of(CHARACTER_ID, 3));
        var retry = ImageTask.builder().refImageUrls(JSON.write(List.of(CHARACTER)))
                .inputMediaVersions(Map.of(CHARACTER_ID, 4)).build();

        assertThatThrownBy(() -> policy().enforce(retry, context))
                .isInstanceOfSatisfying(GenerationJobException.class,
                        failure -> assertThat(failure.getErrorCode()).isEqualTo("GENERATION_INPUT_CHANGED"));
    }

    @Test
    void acceptsFirstGenerationAndTextOnlyTasksWithoutInventingReferenceImages() {
        assetRun();
        var first = ImageTask.builder().prompt("没有参考图的初始形象").build();
        policy().enforce(first, context);
        assertThat(first.getInputMediaVersions()).isNull();

        pinned(null, Map.of());
        policy().enforce(first, context);
        assertThat(first.getInputMediaVersions()).isEmpty();
        assertThat(first.getRefImageUrls()).isNull();
    }

    @Test
    void missingPersistedInputDoesNotTurnIntoAFirstAttempt() {
        assetRun();
        when(submissions.selectAssetImageReferenceSnapshot("scope", "asset-run"))
                .thenReturn(new AssetImageReferenceSnapshot("first-submission", null));
        assertThatThrownBy(() -> policy().enforce(ImageTask.builder().build(), context))
                .hasMessageContaining("缺少首次生成的输入快照");

        pinned(JSON.write(List.of(CHARACTER)), null);
        assertThatThrownBy(() -> policy().enforce(ImageTask.builder().build(), context))
                .hasMessageContaining("缺少首次生成的输入快照");
    }

    @Test
    void ordinaryAssistantAndWorkbenchCanStartDifferentImageRequests() {
        when(runs.selectByRunId(context.agentRunId())).thenReturn(AgentRun.builder()
                .agentType("ai_media").deadlineAt(LocalDateTime.now().plusMinutes(10)).build());
        policy().enforce(ImageTask.builder().refImageUrls(JSON.write(List.of(OTHER))).build(), context);
        policy().enforce(ImageTask.builder().build(), ResourceCommandContext.user(context.actor(), null));

        verifyNoInteractions(submissions, requests);
    }
}
