package com.stonewu.fusion.service.generation.audio;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.stonewu.fusion.entity.generation.AudioTask;
import com.stonewu.fusion.service.ai.ApiConfigService;
import com.stonewu.fusion.service.ai.google.GoogleNativeApi;
import com.stonewu.fusion.service.ai.google.GoogleLiveMusicClient;
import com.stonewu.fusion.service.ai.google.GoogleAudioProtocol;
import com.stonewu.fusion.service.canvas.CanvasJson;
import com.stonewu.fusion.service.generation.GenerationModelCapabilityService;
import com.stonewu.fusion.service.media.MediaInputTransport;
import com.stonewu.fusion.service.media.MediaIntakeService;
import com.stonewu.fusion.service.media.MediaIoBoundary;
import com.stonewu.fusion.service.media.MediaResourceQueries;
import com.stonewu.fusion.service.media.model.MediaResource;
import org.junit.jupiter.api.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.annotation.EnableTransactionManagement;
import org.springframework.transaction.support.TransactionTemplate;

import javax.sql.DataSource;
import java.io.InputStream;
import java.util.Base64;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** 保存已收到的音频不能建立空事务同步作用域，也不能再次调用模型。 */
class AudioTaskRunnerTests {
    @Configuration
    @EnableTransactionManagement
    static class Transactions {
    }

    @Test
    void savingAudioKeepsMediaIoOutsideTransactionSynchronization() throws Exception {
        try (var context = new AnnotationConfigApplicationContext()) {
            context.register(Transactions.class);
            var manager = new DataSourceTransactionManager(mock(DataSource.class));
            context.registerBean(PlatformTransactionManager.class, () -> manager);
            var google = mock(GoogleNativeApi.class);
            var live = mock(GoogleLiveMusicClient.class);
            var intake = mock(MediaIntakeService.class);
            var resource = mock(MediaResource.class);
            when(resource.id()).thenReturn("9e90f72c-85e0-42db-8040-bb01863a53ac");
            when(intake.uploadChecked(any(InputStream.class), anyString(), anyString(), anyString(), isNull(), any(), anySet()))
                    .thenAnswer(call -> {
                        MediaIoBoundary.requireOutsideTransaction();
                        assertThat(call.<InputStream>getArgument(0).readNBytes(4)).containsExactly((byte) 'R', (byte) 'I', (byte) 'F', (byte) 'F');
                        return resource;
                    });
            var json = new CanvasJson(new ObjectMapper().findAndRegisterModules());
            context.registerBean(AudioTaskRunner.class, () -> new AudioTaskRunner(google, live,
                    mock(ApiConfigService.class), mock(GenerationModelCapabilityService.class),
                    mock(MediaInputTransport.class), mock(MediaResourceQueries.class), intake, json,
                    mock(AudioTaskRepository.class)));
            context.refresh();
            var task = new AudioTask();
            task.setId("saved-output");
            task.setUserId(1L);
            task.setOutputJson(json.write(new AudioTaskData.Received("normalized", json.write(
                    new GoogleAudioProtocol.Output("", List.of(new GoogleAudioProtocol.Audio(
                            Base64.getEncoder().encodeToString(new byte[48000]), "audio/pcm;rate=24000")))))));
            var runner = context.getBean(AudioTaskRunner.class);
            assertThat(runner.save(task).audio()).containsExactly("media://9e90f72c-85e0-42db-8040-bb01863a53ac");
            verifyNoInteractions(google, live);
            var synchronization = new TransactionTemplate(manager);
            synchronization.setPropagationBehaviorName("PROPAGATION_NEVER");
            assertThatThrownBy(() -> synchronization.execute(status -> runner.save(task)))
                    .hasMessageContaining("同步作用域之外");
        }
    }
}
