package com.stonewu.fusion.service.media;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.stonewu.fusion.repository.media.MediaResourceRepository;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.canvas.model.CanvasValueType;
import com.stonewu.fusion.service.media.model.MediaException;
import com.stonewu.fusion.service.media.model.MediaFacts;
import com.stonewu.fusion.service.media.model.MediaKind;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.resource.ResourceRestorePlanner.Change;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Map;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

class MediaRestoreValidationTests {
    private final ObjectMapper mapper=new ObjectMapper();
    private final MediaResourceRepository repository=mock(MediaResourceRepository.class);
    private final MediaAccessPolicy access=mock(MediaAccessPolicy.class);
    private final MediaRestoreValidation validation=new MediaRestoreValidation(repository,access);
    private final CanvasActor actor=new CanvasActor(8,null);
    private final String id="7b8e3e23-3244-45c8-8cb5-2b779d3db533";

    @Test void previewAndSubmissionBothCheckCurrentPermissionsForTheProposedMedia() {
        var resource=resource(MediaKind.IMAGE); prepare(resource);
        var changes=List.of(change(CanvasValueType.PROJECT,mapper.createObjectNode().put("coverUrl","media://"+id)));
        validation.validate(changes,actor,false);
        doThrow(new MediaException(403,"MEDIA_ACCESS_DENIED","媒体访问已撤销")).when(access).ready(resource,actor);
        assertThatThrownBy(() -> validation.validate(changes,actor,true)).isInstanceOfSatisfying(MediaException.class,error -> assertThat(error.getCode()).isEqualTo(403));
        verify(repository).get(id,true);
    }
    @Test void explicitSourceGenerationTypeMustMatchTheRestoredAssetMedia() {
        prepare(resource(MediaKind.IMAGE));
        var body=mapper.createObjectNode(); body.putObject("properties").put("sourceMediaUrl","media://"+id).put("sourceGenerationMode","video");
        assertThatThrownBy(() -> validation.validate(List.of(change(CanvasValueType.ASSET,body)),actor,false)).hasMessageContaining("类型不一致");
    }
    @Test void allBusinessMediaOwnersAreCheckedIncludingEpisodeAudio() {
        var fields=Map.of(CanvasValueType.PROJECT,"coverUrl",CanvasValueType.ASSET,"coverUrl",CanvasValueType.ASSET_ITEM,"imageUrl",CanvasValueType.SHOT,"firstFrameImageUrl");
        prepare(resource(MediaKind.IMAGE));
        for(var field:fields.entrySet()) validation.validate(List.of(change(field.getKey(),mapper.createObjectNode().put(field.getValue(),"media://"+id))),actor,false);
        prepare(resource(MediaKind.AUDIO));
        assertThatCode(() -> validation.validate(List.of(change(CanvasValueType.STORYBOARD_EPISODE,mapper.createObjectNode().put("bgmAudioUrl","media://"+id))),actor,false)).doesNotThrowAnyException();
    }
    @Test void removingACreatedViewOfBusinessContentDoesNotRequireOrDeleteItsPrivateFile() {
        validation.validate(List.of(change(CanvasValueType.ASSET,mapper.createObjectNode().put("coverUrl","media://"+id).put("deleted",true))),actor,false);
        verifyNoInteractions(repository,access);
    }
    private void prepare(MediaResource resource) {
        var row=new MediaResourceRepository.Row(resource,"hash","blob",1,null,null);
        when(repository.get(id,false)).thenReturn(row); when(repository.get(id,true)).thenReturn(row);
    }
    private MediaResource resource(MediaKind kind) {
        return new MediaResource(id,null,1,8,8,"历史文件",new MediaFacts(kind,"application/octet-stream",10,"a".repeat(64),null,null,null,false),"READY",1,null,null,null);
    }
    private Change change(CanvasValueType type,ObjectNode proposed) {
        return new Change(type,1,1L,2,mapper.createObjectNode(),proposed,List.of(),false);
    }
}
