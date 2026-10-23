package com.stonewu.fusion.service.generation.job;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.generation.ImageTask;
import com.stonewu.fusion.entity.generation.VideoTask;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.generation.ReferenceImageTransportService;
import com.stonewu.fusion.service.media.MediaAccessPolicy;
import com.stonewu.fusion.service.media.MediaReference;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaResource;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class GenerationInputVersionsTests {
    private final MediaResourceQueries resources = mock(MediaResourceQueries.class);
    private final MediaAccessPolicy access = mock(MediaAccessPolicy.class);
    private final ReferenceImageTransportService references = mock(ReferenceImageTransportService.class);
    private final GenerationInputVersions versions = new GenerationInputVersions(resources, access, references);
    private final CanvasActor actor = new CanvasActor(31L, null);
    private final ObjectMapper json = new ObjectMapper();

    @Test void freezesReferenceImagesAndNestedWorkflowMediaWithoutDroppingExistingPins() {
        String original = registered(3), workflow = registered(7);
        var task = ImageTask.builder().refImageUrls("references").build();
        when(references.parseJsonInputs("references", "refImageUrls"))
                .thenReturn(List.of(MediaReference.of(original)));
        task.setInputMediaVersions(Map.of(original, 3));
        var inputs = json.createObjectNode();
        inputs.putArray("referenceImages").add(MediaReference.of(original)).add(MediaReference.of(workflow));
        inputs.put("negativePrompt", "不模糊");
        task.setWorkflowInputs(inputs);

        versions.freeze(task, actor);

        assertThat(task.getInputMediaVersions()).containsExactlyInAnyOrderEntriesOf(Map.of(original, 3, workflow, 7));
        assertThat(inputs.path("referenceImages").get(1).asText()).isEqualTo(MediaReference.of(workflow));
    }

    @Test void doesNotSilentlyAdvanceAPinnedVersionWhenMediaChangesDuringAdmission() {
        String id = UUID.randomUUID().toString();
        when(resources.get(id, actor)).thenReturn(resource(id, 3), resource(id, 4));
        var task = ImageTask.builder().refImageUrls("references").build();
        when(references.parseJsonInputs("references", "refImageUrls")).thenReturn(List.of(MediaReference.of(id)));
        task.setInputMediaVersions(Map.of(id, 3));

        assertThatThrownBy(() -> versions.freeze(task, actor)).isInstanceOf(GenerationJobException.class)
                .hasMessageContaining("冻结输入期间");
        assertThat(task.getInputMediaVersions()).containsEntry(id, 3);
    }

    @Test void freezesVideoFramesAndAllReferenceMediaKinds() {
        String first = registered(1), last = registered(2), image = registered(3), video = registered(4), audio = registered(5);
        var task = VideoTask.builder().firstFrameImageUrl(MediaReference.of(first)).lastFrameImageUrl(MediaReference.of(last))
                .referenceImageUrls("images").referenceVideoUrls("videos").referenceAudioUrls("audios").build();
        when(references.parseJsonInputs("images", "referenceImageUrls")).thenReturn(List.of(MediaReference.of(image)));
        when(references.parseJsonInputs("videos", "referenceVideoUrls")).thenReturn(List.of(MediaReference.of(video)));
        when(references.parseJsonInputs("audios", "referenceAudioUrls")).thenReturn(List.of(MediaReference.of(audio)));

        versions.freeze(task, actor);

        assertThat(task.getInputMediaVersions()).containsExactlyInAnyOrderEntriesOf(Map.of(first, 1, last, 2, image, 3, video, 4, audio, 5));
    }

    @Test void rejectsChangedOrMissingSnapshotsBeforeProviderPreparation() {
        String id = registered(9);
        assertThatThrownBy(() -> versions.validate(Map.of(id, 8), actor))
                .isInstanceOf(GenerationJobException.class).hasMessageContaining("未继续提交供应商");
        assertThatThrownBy(() -> versions.validate(null, actor))
                .isInstanceOf(GenerationJobException.class).hasMessageContaining("缺少输入媒体版本快照");
        versions.validate(Map.of(), actor);
    }

    private String registered(int version) {
        String id = UUID.randomUUID().toString();
        when(resources.get(id, actor)).thenReturn(resource(id, version));
        return id;
    }

    private MediaResource resource(String id, int version) {
        return new MediaResource(id, 1L, 1, actor.userId(), actor.userId(), "输入媒体", null, "READY", version, null, null, null);
    }
}
