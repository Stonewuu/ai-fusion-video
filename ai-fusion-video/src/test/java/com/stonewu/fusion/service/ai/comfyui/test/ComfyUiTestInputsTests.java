package com.stonewu.fusion.service.ai.comfyui.test;

import com.stonewu.fusion.entity.ai.ComfyUiWorkflow;
import com.stonewu.fusion.entity.ai.ComfyUiWorkflowVersion;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiInputBinding;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowDocumentService;
import com.stonewu.fusion.service.ai.comfyui.ComfyUiWorkflowRenderer;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.MediaInputTransport;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaFacts;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.media.model.MediaResource;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class ComfyUiTestInputsTests {
    private static final String AUDIO_ID = "00000000-0000-0000-0000-000000000001";
    private static final String VIDEO_ID = "00000000-0000-0000-0000-000000000002";
    private final CanvasActor actor = new CanvasActor(1, null);
    private final ComfyUiWorkflowDocumentService documents = mock(ComfyUiWorkflowDocumentService.class);
    private final MediaResourceQueries media = mock(MediaResourceQueries.class);
    private final MediaInputTransport transport = mock(MediaInputTransport.class);
    private final ComfyUiTestInputs inputs = new ComfyUiTestInputs(documents,
            mock(ComfyUiWorkflowRenderer.class), media, transport);
    private final ComfyUiWorkflow workflow = ComfyUiWorkflow.builder().id(7L).modelType(3).build();
    private final ComfyUiWorkflowVersion version = ComfyUiWorkflowVersion.builder()
            .apiWorkflowJson("{}").inputBindingsJson("{}").build();

    @Test
    void audioAndVideoInputsFreezeVersionsAndResolveWithTheirOwnMediaKinds() {
        when(documents.parseInputBindings(3, "{}", "{}")).thenReturn(List.of(
                new ComfyUiInputBinding("referenceAudios", "1", "audio", "uploaded_audio", 0),
                new ComfyUiInputBinding("referenceVideos", "2", "file", "uploaded_video", 0)));
        when(media.get(AUDIO_ID, actor)).thenReturn(resource(AUDIO_ID, MediaKind.AUDIO, 3));
        when(media.get(VIDEO_ID, actor)).thenReturn(resource(VIDEO_ID, MediaKind.VIDEO, 4));
        when(transport.dataUri("media://" + AUDIO_ID, actor, MediaKind.AUDIO, MediaIntakeService.MAX_UPLOAD_BYTES))
                .thenReturn("data:audio/wav;base64,AQID");
        when(transport.dataUri("media://" + VIDEO_ID, actor, MediaKind.VIDEO, MediaIntakeService.MAX_UPLOAD_BYTES))
                .thenReturn("data:video/mp4;base64,AQID");
        var snapshot = inputs.freeze(workflow, version, Map.of("referenceAudios", List.of("media://" + AUDIO_ID),
                "referenceVideos", List.of("media://" + VIDEO_ID)), actor);
        assertThat(snapshot.mediaVersions()).containsExactlyInAnyOrderEntriesOf(Map.of(AUDIO_ID, 3, VIDEO_ID, 4));
        assertThat(inputs.resolve(snapshot, actor)).containsEntry("referenceAudios", List.of("data:audio/wav;base64,AQID"))
                .containsEntry("referenceVideos", List.of("data:video/mp4;base64,AQID"));
        verify(transport).dataUri("media://" + AUDIO_ID, actor, MediaKind.AUDIO, MediaIntakeService.MAX_UPLOAD_BYTES);
        verify(transport).dataUri("media://" + VIDEO_ID, actor, MediaKind.VIDEO, MediaIntakeService.MAX_UPLOAD_BYTES);
    }

    @Test
    void changedAudioInputVersionCannotBeResolvedForTheOldTestSnapshot() {
        when(documents.parseInputBindings(3, "{}", "{}")).thenReturn(List.of(
                new ComfyUiInputBinding("referenceAudios", "1", "audio", "uploaded_audio", 0)));
        when(media.get(AUDIO_ID, actor)).thenReturn(resource(AUDIO_ID, MediaKind.AUDIO, 3));
        var snapshot = inputs.freeze(workflow, version, Map.of("referenceAudios", List.of("media://" + AUDIO_ID)), actor);
        when(media.get(AUDIO_ID, actor)).thenReturn(resource(AUDIO_ID, MediaKind.AUDIO, 4));
        assertThatThrownBy(() -> inputs.resolve(snapshot, actor)).isInstanceOf(MediaException.class)
                .hasMessageContaining("版本已变化");
        verifyNoInteractions(transport);
    }

    @Test
    void uploadedAudioBindingRejectsVideoReferences() {
        when(documents.parseInputBindings(3, "{}", "{}")).thenReturn(List.of(
                new ComfyUiInputBinding("referenceAudios", "1", "audio", "uploaded_audio", 0)));
        when(media.get(VIDEO_ID, actor)).thenReturn(resource(VIDEO_ID, MediaKind.VIDEO, 4));
        assertThatThrownBy(() -> inputs.freeze(workflow, version,
                Map.of("referenceAudios", List.of("media://" + VIDEO_ID)), actor))
                .isInstanceOf(MediaException.class).hasMessageContaining("媒体类型与绑定不匹配");
    }

    private MediaResource resource(String id, MediaKind kind, int version) {
        var facts = new MediaFacts(kind, kind == MediaKind.AUDIO ? "audio/wav" : "video/mp4", 3, "test-hash",
                null, null, 1000L, kind == MediaKind.AUDIO);
        return new MediaResource(id, 7L, 1, 1, 1, "验收媒体", facts, "READY", version, null, null, null);
    }
}
