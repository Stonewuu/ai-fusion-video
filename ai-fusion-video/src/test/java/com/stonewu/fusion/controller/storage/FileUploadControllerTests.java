package com.stonewu.fusion.controller.storage;

import com.stonewu.fusion.common.BusinessException;
import com.stonewu.fusion.entity.ai.AiModel;
import com.stonewu.fusion.service.ai.AiModelService;
import com.stonewu.fusion.service.canvas.CanvasActorResolver;
import com.stonewu.fusion.service.canvas.model.CanvasActor;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.model.MediaResource;
import com.stonewu.fusion.service.system.SystemConfigService;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;
import java.util.List;
import java.util.Map;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class FileUploadControllerTests {
    private final MediaIntakeService intake = mock(MediaIntakeService.class);
    private final CanvasActorResolver actors = mock(CanvasActorResolver.class);
    private final SystemConfigService system = mock(SystemConfigService.class);
    private final AiModelService models = mock(AiModelService.class);
    private final FileUploadController controller = new FileUploadController(intake, actors, system, models);
    private final MockMultipartFile file = new MockMultipartFile("file", "sample.png", "image/png", new byte[]{1});
    private static final String ID = "63c234b0-e0fd-4f65-8270-b26e282de689";

    @Test void uploadsIntoRegistryWithAuthenticatedIdentityAndStableReference() throws Exception {
        when(actors.current()).thenReturn(new CanvasActor(4L, 8L));
        when(intake.uploadChecked(any(), eq("sample.png"), eq("image/png"), eq("request-a"), isNull(), any(), anySet()))
                .thenAnswer(call -> {
                    assertThat(call.getArgument(5).toString()).contains("userId=4", "teamId=8");
                    return new MediaResource(ID, null, 2, 8, 4, "sample.png", null, "READY", 1, null, null, null);
                });
        assertThat(controller.upload(file, "request-a").getData()).isEqualTo("media://" + ID);
        verifyNoInteractions(system, models);
    }
    @Test void urlInputReturnsPrivateReferenceAfterModelValidation() throws Exception {
        when(models.getById(7L)).thenReturn(model(List.of("url")));
        when(system.getPublicResourceBaseUrl()).thenReturn("https://media.example.test");
        when(actors.current()).thenReturn(new CanvasActor(4L, null));
        when(intake.uploadChecked(any(), anyString(), anyString(), anyString(), isNull(), any(), anySet()))
                .thenReturn(new MediaResource(ID, null, 1, 4, 4, "sample.png", null, "READY", 1, null, null, null));
        assertThat(controller.uploadAssistantInput(file, 7L, "request-b", "url").getData()).isEqualTo("media://" + ID);
    }
    @Test void rejectsUnsupportedTransportBeforeStoring() {
        when(models.getById(7L)).thenReturn(model(List.of("base64")));
        assertThatThrownBy(() -> controller.uploadAssistantInput(file, 7L, "request-b", "url"))
                .isInstanceOf(BusinessException.class).hasMessageContaining("不支持 image 的 url 输入");
        verifyNoInteractions(intake);
    }
    @Test void base64PreviewDoesNotRequirePublicAddress() throws Exception {
        when(models.getById(7L)).thenReturn(model(List.of("base64")));
        when(actors.current()).thenReturn(new CanvasActor(4L, null));
        when(intake.uploadChecked(any(), anyString(), anyString(), anyString(), isNull(), any(), anySet()))
                .thenReturn(new MediaResource(ID, null, 1, 4, 4, "sample.png", null, "READY", 1, null, null, null));
        assertThat(controller.uploadAssistantInput(file, 7L, "request-b", "base64").getData()).isEqualTo("media://" + ID);
        verifyNoInteractions(system);
    }
    @Test void rejectsUrlUploadWithoutPublicGatewayEvenWithObjectStorage() {
        when(models.getById(7L)).thenReturn(model(List.of("url")));
        assertThatThrownBy(() -> controller.uploadAssistantInput(file, 7L, "request-b", "url"))
                .hasMessageContaining("后端资源公网地址");
        verifyNoInteractions(intake);
    }
    private AiModel model(List<String> transports) {
        return AiModel.builder().modelType(1).status(1).multimodalInputTypes(List.of("image"))
                .multimodalInputTransports(Map.of("image", transports)).build();
    }
}
